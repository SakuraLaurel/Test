from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
import kornia.feature as KF
from glob import glob
import pandas as pd
import numpy as np
import cv2
import gc

from nets.aliked import ALIKED
import matplotlib.pyplot as plt
from pickle import dump, load


class Category(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def query(self, scene):
        return self.data['categories'][self.data['scene'] == scene].item()

    def transparent(self, scene):
        return 'transparent' in self.query(scene)

    def use_crops(self, scene):
        # 只有不透明又存在对称区域的，才需要使用crop匹配
        cats = self.query(scene)
        return ('symmetr' not in cats) and (not self.transparent(scene))


class Img(object):
    def __init__(self, path):
        # 高度×宽度×通道
        self.data = cv2.imread(path, cv2.IMREAD_COLOR)

    @property
    def shape(self):
        return self.data.shape

    @property
    def h(self):
        return self.shape[0]

    @property
    def w(self):
        return self.shape[1]

    def resize(self, size, square=False):
        # ssim的计算只能使用两张形状一致的图像，这与是否旋转是冲突的，所以有时需要拓展为方形图片
        def rot(i):
            return np.transpose(i, (1, 0, 2)) if self.h < self.w else i

        data = rot(self.data)
        if square:
            h, w, c = data.shape
            img = np.zeros((h, h, c), dtype=np.uint8)
            start = int((h - w) / 2)
            img[:, start:start + w] = np.float32(data)
            data = img
        width = int(size * data.shape[1] / data.shape[0])
        data = cv2.resize(data,(width, size))  # 传入的shape是宽度×高度
        return rot(data)


class Imgs(object):
    def __init__(self, dir):
        self.data = list(map(Img, sorted(glob(f"{dir}/*"))[:]))

    def resize(self, size, square=False):
        return [img.resize(size, square=square) for img in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

def rot90(imgs):
    res = [imgs]
    for r in range(1, 4):
        res.append([torch.rot90(i, r, (1, 2)) for i in imgs])
    return res

def batch_divide(func, n, batch_size=1):
    if n <= batch_size:
        return func(slice(0, n))
    times = int(n / batch_size)
    start = n - times * batch_size
    res = []
    if start != 0:
        res.append(func(slice(0, start)))
    for i in range(times):
        index = start + i * batch_size
        res.append(func(slice(index, index + batch_size)))
    return torch.cat(res)


class Ssim(object):
    s = StructuralSimilarityIndexMeasure(
        data_range=255., reduction='none').cuda()
    size = 2048

    @classmethod
    def score(cls, i1s, i2s):
        def f(s):
            with torch.inference_mode():
                res = cls.s(i1s[s].cuda(), i2s[s].cuda()).cpu()
                if res.ndim == 0:
                    res = torch.tensor([res])
                return res
        return batch_divide(f, i1s.shape[0])

    @classmethod
    def rot(cls, imgs):
        # (旋转角度, batch, channel, height, width)
        imgs4 = torch.tensor(np.array(rot90(imgs)), dtype=torch.float32)
        n = len(imgs)
        i1s = imgs4[0, 0:1].expand(4 * (n - 1), -1, -1, -1)
        i2s = imgs4[:, 1:].reshape((4 * (n - 1), ) + imgs4.shape[2:])
        scores = cls.score(i1s, i2s).reshape((4, n-1))
        min_diff_idx = torch.cat([torch.tensor([0]), torch.argmax(scores, 0)])
        return imgs4[min_diff_idx, torch.arange(n)]

    @classmethod
    def flows(cls, imgs):
        # 拓展为(batch, channel, height, width)。
        # 根据SSIM将影像进行旋转校正，但现有数据不存在旋转问题，影像也不是方形的；填充0会影响计算结果，不填充无法计算。为了与原作保持一致，可替换为以下代码。
        # imgs = cls.rot(torch.tensor(np.transpose(imgs.resize(cls.size, square=True), (0, 3, 1, 2)), dtype=torch.float32))
        # 与上一行互斥
        resized = np.transpose(imgs.resize(cls.size), (0, 3, 1, 2))
        imgs = torch.tensor(resized, dtype=torch.float32)
        n = len(imgs)
        ssim_flows = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n - 1):
            criteria = imgs[i:i + 1].expand(n - i - 1, -1, -1, -1)
            ssim_flows[i, i + 1:] = 1 - cls.score(criteria, imgs[i + 1:])
            ssim_flows[i + 1:, i] = ssim_flows[i, i + 1:]
        return ssim_flows


class Aliked(object):
    aliked_extractor = ALIKED(model_name="aliked-n16").cuda().eval()

    @classmethod
    def extract(cls, img):
        res = cls.aliked_extractor(img[None].cuda())
        # 原始区间[-1, 1]，转换到[0, h-1]和[0, w-1]区间内
        wh = (torch.tensor(img.shape[2:0:-1]) - 1).reshape((1, 2)).cuda()
        res['keypoints'] = wh * (res['keypoints'][0] + 1) / 2.0
        res['descriptors'] = res['descriptors'][0]
        return res

class LightGlue(object):
    matcher = KF.LightGlueMatcher(feature_name="aliked", params={
        "filter_threshold": 0.2,
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True
    }).cuda().eval()
    sizes = (1024, 1280, 1600)

    @classmethod
    def resize(cls, imgs, size_i):
        resized = imgs.resize(cls.sizes[size_i])
        transposed = [np.transpose(i / 255., (2, 0, 1)) for i in resized]
        return [torch.tensor(i, dtype=torch.float32) for i in transposed]

    @classmethod
    def match(cls, p1s, p2s):
        k1s = p1s['keypoints']
        k2s = p2s['keypoints']
        with torch.inference_mode():
            desc1 = p1s['descriptors']
            desc2 = p2s['descriptors']
            lafs1 = KF.laf_from_center_scale_ori(k1s[None])
            lafs2 = KF.laf_from_center_scale_ori(k2s[None])
            _, indices = cls.matcher(desc1, desc2, lafs1, lafs2)
            mk1s = k1s[indices[:, 0]].cpu().numpy()
            mk2s = k2s[indices[:, 1]].cpu().numpy()
            try:
                _, inliers = cv2.findFundamentalMat(
                    mk1s, mk2s, cv2.USAC_MAGSAC, ransacReprojThreshold=5, confidence=0.9999, maxIters=50000)
                inliers = inliers.ravel() > 0
                return len(inliers)
            except BaseException:
                return 0

    @classmethod
    def rot(cls, imgs):
        imgs4 = rot90(cls.resize(imgs, 0))
        rots = [0 for _ in imgs]
        for i in range(1, len(imgs)):
            num_matches = 0
            for j in range(len(cls.sizes)):
                p1s = Aliked.extract(imgs4[0][0])
                p2s = Aliked.extract(imgs4[j][i])
                res = cls.match(p1s, p2s)
                if res > num_matches:
                    rots[i] = j
                    num_matches = res
        return rots

    @classmethod
    def flows(cls, imgs):
        n = len(imgs)
        imgs3n = [cls.resize(imgs, i) for i in range(len(cls.sizes))]
        imgsn3 = [[None for _ in cls.sizes] for _ in range(n)]
        for i, rot in enumerate(cls.rot(imgs)):
            for j, imgsn in enumerate(imgs3n):
                imgsn3[i][j] = torch.rot90(imgsn[i], rot, (1, 2))
        res = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            res[i, i] = 0
            for j in range(i + 1, n):
                v = sum([cls.match(Aliked.extract(i1), Aliked.extract(i2))
                        for i1, i2 in zip(imgsn3[i], imgsn3[j])])
                res[i, j] = 1e2 / v
                res[j, i] = res[i, j]
        return res
    

class Tsp(object):
    @classmethod
    def solve(cls, distance_matrix, start_idx):
        manager = pywrapcp.RoutingIndexManager(
            distance_matrix.shape[0], 1, start_idx)
        routing = pywrapcp.RoutingModel(manager)
        callback = routing.RegisterTransitCallback(
            lambda i, j: distance_matrix[manager.IndexToNode(i), manager.IndexToNode(j)])
        routing.SetArcCostEvaluatorOfAllVehicles(callback)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        assignment = routing.SolveWithParameters(search_parameters)
        index = routing.Start(0)
        res = []
        while not routing.IsEnd(index):
            res.append(manager.IndexToNode(index))
            index = assignment.Value(routing.NextVar(index))
        return np.array(res)

    @classmethod
    def normalize(cls, distance_matrix):
        order_idxs = np.empty(distance_matrix.shape, dtype=np.int32)
        for i in range(order_idxs.shape[0]):
            order_idxs[i] = cls.solve(distance_matrix, i)
        neighbors = np.zeros(distance_matrix.shape, dtype=np.int32)
        for i in order_idxs:
            left = np.concatenate((i[1:], i[:1]), axis=0)
            neighbors[i, left] += 100
            neighbors[left, i] += 100
        neighbors[neighbors == 0] = 1
        res = 1e8 / neighbors
        np.fill_diagonal(res, 0)
        return res

    @classmethod
    def find(cls, distance_matrix):
        res = np.zeros(np.array(distance_matrix.shape) + 1)
        res[1:, 1:] = cls.normalize(distance_matrix)
        return cls.solve(res, 0)[1:] - 1


def get_rmats(n):
    theta = 2 * torch.pi / n
    return [cv2.Rodrigues(np.array([0, 0, 1]) * theta * i)[0]
            for i in range(n)]


def main():
    scenes = ["transp_obj_glass_cylinder"]
    category = Category("train/categories.csv")
    for scene in scenes:
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"{scene=} {category.query(scene)=} {category.transparent(scene)=} {category.use_crops(scene)=}")
        # imgs = Imgs(f"train/{scene}/images")
        if category.transparent(scene):
            # ssim_flows = Ssim.flows(imgs)
            # matching_flows = LightGlue.flows(imgs)
            # with open("flows.pkl", "wb") as f:
                # dump((ssim_flows, matching_flows), f)
            with open("flows.pkl", "rb") as f:
                ssim_flows, matching_flows = load(f)
            distance_matrix = np.int32((ssim_flows + matching_flows) * 1e6)
            print(Tsp.find(distance_matrix))
            # rmats(len(imgs))

    # results_df = pd.DataFrame(columns=['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector'])


if __name__ == "__main__":
    main()