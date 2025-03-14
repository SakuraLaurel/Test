import pandas as pd
from nets.aliked import ALIKED
import kornia.feature as KF
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
import gc
import torch
import cv2
from glob import glob
import matplotlib.pyplot as plt

class Category(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def query(self, scene):
        return self.data['categories'][self.data['scene']==scene].item()

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
        data = cv2.resize(data, (int(size * data.shape[1] / data.shape[0]), size))  # 传入的shape是宽度×高度
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

def batch_divide(func, n, batch_size=1):
    if n <= batch_size:
        return func(slice(0, n))
    times = int(n / batch_size)
    start = n - times * batch_size
    res = []
    if start != 0:
        res.append(func(slice(0, start)))
    for i in range(times):
        res.append(func(slice(start + i * batch_size, start + (i+1) * batch_size)))
    return torch.cat(res)
    

class Ssim(object):
    s = StructuralSimilarityIndexMeasure(data_range=255., reduction='none').cuda()
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
        n = len(imgs)
        # (batch, 旋转角度, channel, height, width)
        imgs4 = torch.empty((n, 4) + imgs.shape[1:])
        imgs4[:, 0] = imgs
        for r in range(1, 4):
            imgs4[:, r] = torch.rot90(imgs, r, (2, 3))
        i1s = imgs4[0:1, 0].expand(4*(n-1), -1, -1, -1)
        i2s = imgs4[1:].reshape((4*(n-1), ) + imgs4.shape[2:])
        scores = cls.score(i1s, i2s).reshape((n-1, 4))
        min_diff_idx = torch.cat([torch.tensor([0]), torch.argmax(scores, 1)])
        return imgs4[torch.arange(n), min_diff_idx]

    @classmethod
    def flows(cls, imgs):
        # 拓展为(batch, channel, height, width)。
        # 根据SSIM将影像进行旋转校正，但现有数据不存在旋转问题，影像也不是方形的；填充0会影响计算结果，不填充无法计算。为了与原作保持一致，可替换为以下代码。
        # imgs = cls.rot(torch.tensor(np.transpose(imgs.resize(cls.size, square=True), (0, 3, 1, 2)), dtype=torch.float32))
        # 与上一行互斥
        imgs = torch.tensor(np.transpose(imgs.resize(cls.size), (0, 3, 1, 2)), dtype=torch.float32)
        n = len(imgs)
        ssim_flows = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n-1):
            ssim_flows[i, i+1:] = 1 - cls.score(imgs[i:i+1].expand(n-i-1,-1,-1,-1), imgs[i+1:])
            ssim_flows[i+1:, i] = ssim_flows[i, i+1:]
        return ssim_flows


class Lightglue(object):
    matcher = KF.LightGlueMatcher(feature_name="aliked", params={
        "filter_threshold": 0.2,
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True
    }).cuda().eval()
    
    @classmethod
    def match(cls, p1s, p2s):
        k1s = p1s['keypoints']
        k2s = p2s['keypoints']
        with torch.inference_mode():
            _, indices = cls.matcher(p1s['descriptors'], p2s['descriptors'], KF.laf_from_center_scale_ori(k1s[None]), KF.laf_from_center_scale_ori(k2s[None]))
            mk1s = k1s[indices[:, 0]].cpu().numpy()
            mk2s = k2s[indices[:, 1]].cpu().numpy()
            try:
                _, inliers = cv2.findFundamentalMat(mk1s, mk2s, cv2.USAC_MAGSAC, ransacReprojThreshold=5, confidence=0.9999, maxIters=50000)
                inliers = inliers.ravel() > 0
                return len(inliers)
            except:
                return 0

class Aliked_LightGlue(object):
    sizes = (1024, 1280, 1600)
    aliked_extractor = ALIKED(model_name="aliked-n16").cuda().eval()

    @classmethod
    def resize(cls, imgs, size_i):
        return [torch.tensor(np.transpose(i / 255., (2, 0, 1)), dtype=torch.float32) for i in imgs.resize(cls.sizes[size_i])]

    @classmethod
    def extract(cls, img):
        res = cls.aliked_extractor(img[None].cuda())
        # 原始区间[-1, 1]，转换到[0, h-1]和[0, w-1]区间内
        wh = (torch.tensor(img.shape[2:0:-1]) - 1).reshape((1,2)).cuda()
        res['keypoints'] = wh * (res['keypoints'][0] + 1) / 2.0
        res['descriptors'] = res['descriptors'][0]
        return res

    @classmethod
    def rot(cls, imgs):
        imgs4 = [[torch.rot90(i, rot, (1, 2)) for rot in range(4)] for i in cls.resize(imgs, 0)]
        rots = [0 for _ in imgs]
        for i in range(1, len(imgs)):
            num_matches = 0
            for j in range(4):
                res = Lightglue.match(cls.extract(imgs4[0][0]), cls.extract(imgs4[i][j]))
                if res > num_matches:
                    rots[i] = j
                    num_matches = res
        return rots

    @classmethod
    def flows(cls, imgs):
        n = len(imgs)
        rots = cls.rot(imgs)
        imgs3 = [cls.resize(imgs, i) for i in range(len(cls.sizes))]
        imgs3 = [[torch.rot90(imgs3[i][j], rots[j], (1, 2)) for i in range(len(cls.sizes))] for j in range(n)]
        res = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            res[i, i] = 0
            for j in range(i+1, n):
                v = sum([Lightglue.match(cls.extract(i1), cls.extract(i2)) for i1, i2 in zip(imgs3[i], imgs3[j])])
                res[i, j] = int(1e8 / v)
                res[j, i] = res[i, j]
        return res
    

def get_rmats(n):
    theta = 2 * torch.pi / n
    return [cv2.Rodrigues(np.array([0, 0, 1]) * theta * i)[0] for i in range(n)]


def main():
    scenes = ["transp_obj_glass_cylinder"]
    category = Category("train/categories.csv")
    for scene in scenes:
        
        gc.collect()
        
        # print(f"{scene=} {category.query(scene)=} {category.transparent(scene)=} {category.use_crops(scene)=}")

        imgs = Imgs(f"train/{scene}/images")
        if category.transparent(scene):
            # ssim_flows = Ssim.flows(imgs)
            matching_flows = Aliked_LightGlue.flows(imgs)
            plt.imshow(matching_flows)
            plt.colorbar()
            plt.show()

            Rmats = get_rmats(len(imgs))
    
    # results_df = pd.DataFrame(columns=['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector'])

if __name__ == "__main__":
    main()