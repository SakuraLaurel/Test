import pandas as pd
from nets.aliked import ALIKED
import kornia
import kornia.feature as KF
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
from functools import reduce
import time
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
    
    def resize(self, size, square=True):
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
        self.data = list(map(Img, sorted(glob(f"{dir}/*"))[:10]))

    def resize(self, size):
        return [img.resize(size) for img in self.data]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

class Ssim(object):
    s = StructuralSimilarityIndexMeasure(data_range=255., reduction='none').cuda()
    size = 2048
    batch_size = 5

    @classmethod
    def score(cls, i1s, i2s):
        n = i1s.shape[0]
        if n <= cls.batch_size:
            return cls.s(i1s, i2s).cpu()
        res = torch.zeros((n, ), dtype=torch.float32)
        times = int(n / cls.batch_size)
        start = n - times * cls.batch_size
        if start != 0:
            res[:start] = cls.s(i1s[:start], i2s[:start]).cpu()
        for i in range(times):
            s = slice(start + i*cls.batch_size, start + (i+1)*cls.batch_size)
            res[s] = cls.s(i1s[s], i2s[s]).cpu()
        return res

    @classmethod
    def rot(cls, imgs):
        n = len(imgs)
        # (batch, 旋转角度, channel, height, width)
        imgs4 = torch.empty((n, 4) + imgs.shape[1:])
        imgs4[:, 0] = imgs
        for r in range(1, 4):
            imgs4[:, r] = torch.rot90(imgs, r, (2, 3))
        with torch.inference_mode():
            i1s = imgs4[0:1, 0].expand(4*(n-1), -1, -1, -1).cuda()
            i2s = imgs4[1:].reshape((4*(n-1), ) + imgs4.shape[2:]).cuda()
            scores = cls.score(i1s, i2s).reshape((n-1, 4))
            min_diff_idx = torch.cat([torch.tensor([0]), torch.argmax(scores, 1)])
            return imgs4[torch.arange(n), min_diff_idx]

    @classmethod
    def flows(cls, imgs):
        n = len(imgs)
        # 拓展为(batch, channel, height, width)
        imgs = cls.rot(torch.tensor(np.transpose(imgs.resize(cls.size), (0, 3, 1, 2)), dtype=torch.float32))
        ssim_flows = np.zeros((n, n), dtype=np.float32)
        with torch.inference_mode():
            for i in range(n-1):
                ssim_flows[i, i+1:] = 1 - cls.score(imgs[i:i+1].expand(n-i-1,-1,-1,-1).cuda(), imgs[i+1:].cuda()).numpy()
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
    def match(cls, p1, p2):
        k1 = p1['keypoints']
        k2 = p2['keypoints']
        with torch.inference_mode():
            _, indices = cls.matcher(p1['descriptors'][0], p2['descriptors'][0], KF.laf_from_center_scale_ori(k1), KF.laf_from_center_scale_ori(k2))
            time.sleep(1)
            k1 = k1[0].cpu().numpy()
            k2 = k2[0].cpu().numpy()
            indices = indices.cpu().numpy()
        mk1 = np.float32(k1[indices[..., 0]])
        mk2 = np.float32(k2[indices[..., 1]])
        try:
            _, inliers = cv2.findFundamentalMat(mk1, mk2, cv2.USAC_MAGSAC, ransacReprojThreshold=5, confidence=0.9999, maxIters=50000)
            inliers = inliers.ravel() > 0
            return len(inliers)
        except:
            return 0

class Aliked_LightGlue(object):
    sizes = (1024, 1280, 1600)
    aliked_extractor = ALIKED(model_name="aliked-n16").cuda().eval()

    @classmethod
    def numpy_image_to_torch(cls, img):
        if img.ndim == 3:
            img = img.transpose((2, 0, 1))  # HxWxC to CxHxW
        elif img.ndim == 2:
            img = img[None]
        else:
            raise ValueError(f"Not an image: {img.shape}")
        return torch.tensor(img / 255.0, dtype=torch.float32)[None]

    @classmethod
    def pred(cls, img, size, rot):
        img = cls.numpy_image_to_torch(img.resize(size))
        tensor = torch.rot90(img, rot, (2, 3)).cuda()
        h, w = tensor.shape[2:4]
        hw = torch.tensor([w-1, h-1]).reshape((1,1,1,2)).to(tensor.device)
        res = cls.aliked_extractor(tensor)
        # 原始区间[-1, 1]，转换到[0, h-1]和[0, w-1]区间内
        res['keypoints'] = hw * (torch.stack(res['keypoints']) + 1) / 2.0
        return res


    @classmethod
    def match(cls, p1_f, p2_f):
        rot, num_matches = 0, 0
        for ori in range(4):
            n = Lightglue.match(p1_f(0, 0), p2_f(0, ori))
            if n > num_matches:
                rot = ori
                num_matches = n
        for size_i in range(len(cls.sizes)):
            num_matches += Lightglue.match(p1_f(size_i, 0), p2_f(size_i, rot))
        return num_matches

    @classmethod
    def flows(cls, imgs):
        cache = [[[None for _ in range(4)] for _ in cls.sizes] for _ in imgs]

        def f(img_i):
            def g(size_i, rot):
                if cache[img_i][size_i][rot] is None:
                    cache[img_i][size_i][rot] = cls.pred(imgs[img_i], cls.sizes[size_i], rot)
                return cache[img_i][size_i][rot]
            return g
        
        n = len(imgs)
        res = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            res[i, i] = 0
            for j in range(i+1, n):
                res[i, j] = int(1e8 / cls.match(f(i), f(j)))
                res[j, i] = res[i, j]
        return res
    

def get_rmats(n):
    theta = 2 * torch.pi / n
    return [cv2.Rodrigues(np.array([0, 0, 1]) * theta * i)[0] for i in range(n)]


def main():
    scenes = ["transp_obj_glass_cylinder"]
    category = Category("train/categories.csv")
    for scene in scenes:
        torch.cuda.empty_cache()
        gc.collect()
        
        # print(f"{scene=} {category.query(scene)=} {category.transparent(scene)=} {category.use_crops(scene)=}")

        imgs = Imgs(f"train/{scene}/images")
        if category.transparent(scene):
            ssim_flows = Ssim.flows(imgs)
            # matching_flows = Aliked_LightGlue.flows(imgs)
            plt.imshow(ssim_flows)
            plt.colorbar()
            plt.show()

            Rmats = get_rmats(len(imgs))
    
    # results_df = pd.DataFrame(columns=['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector'])
main()