from transp import Aliked, LightGlue
import numpy as np
import kornia.feature as KF
from glob import glob
import cv2
import torch

def imread(i, size=1024):
    img = cv2.imread(sorted(glob(f"train/transp_obj_glass_cylinder/images/*"))[i], cv2.IMREAD_COLOR)
    def rot(i):
        return np.transpose(i, (1, 0, 2)) if img.shape[0] < img.shape[1] else i
    data = rot(img)
    data = cv2.resize(data, (int(size * data.shape[1] / data.shape[0]), size))
    return rot(data)

def to_tensor(img):
    # 传入的shape是宽度×高度
    return torch.tensor(np.transpose(img / 255.0, (2, 0, 1)), dtype=torch.float32)

def display_lightblue_matches(index1, index2):
    i1 = imread(index1)
    i2 = imread(index2)
    p1s = Aliked.extract(to_tensor(i1))
    p2s = Aliked.extract(to_tensor(i2))
    k1s = p1s['keypoints']
    k2s = p2s['keypoints']
    with torch.inference_mode():
        distances, indices = LightGlue.matcher(p1s['descriptors'], p2s['descriptors'], KF.laf_from_center_scale_ori(k1s[None]), KF.laf_from_center_scale_ori(k2s[None]))
        mk1s = [cv2.KeyPoint(x=i[0], y=i[1], size=1) for i in k1s[indices[:, 0]].cpu().numpy()]
        mk2s = [cv2.KeyPoint(x=i[0], y=i[1], size=1) for i in k2s[indices[:, 1]].cpu().numpy()]
        dmatch = [cv2.DMatch(i, i, int(distances.ravel()[i])) for i in range(len(indices))]
        res = cv2.drawMatches(i1, mk1s, i2, mk2s, dmatch, None)
        cv2.imshow("匹配结果", res)
        cv2.waitKey(0)


display_lightblue_matches(0, 1)