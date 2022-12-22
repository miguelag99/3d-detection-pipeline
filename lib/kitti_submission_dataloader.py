import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline

class KITTIStereoDataloader(object):
    def __init__(self, filepath, split, dynamic_bs=False, kitti2015=False):
        self.dynamic_bs = dynamic_bs
        left_fold = 'image_2/'
        right_fold = 'image_3/'
        calib_fold = 'calib/'
        with open(split, 'r') as f:
            image = list(map(lambda x: x.strip(), f.readlines()))
            # image = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        
        image = sorted(image)

        if kitti2015:
            self.left_test = [filepath + '/' + left_fold + img + '_10.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '_10.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]
        else:
            self.left_test = [filepath + '/' + left_fold + img + '.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.kitti_baseline = 0.54


    def __getitem__(self, item):
        left_img = self.left_test[item]
        right_img = self.right_test[item]
        calib_info = read_calib_file(self.calib_test[item])
        
        # returns only fy*baseline to transform disparity to depth
        if self.dynamic_bs:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * self.kitti_baseline
        
        imgL = Image.open(left_img).convert('RGB')
        imgR = Image.open(right_img).convert('RGB')
        imgL = self.trans(imgL)[None, :, :, :]
        imgR = self.trans(imgR)[None, :, :, :]

        # pad to (384, 1248) if needed
        B, C, H, W = imgL.shape
        top_pad = 0
        right_pad = 0
        if H < 384:
            top_pad = 384 - H
        if W < 1248:
            right_pad = 1248 - W

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)

        filename = self.left_test[item].split('/')[-1][:-4]

        return imgL.float(), imgR.float(), calib.item(), H, W, filename


    def __len__(self):
        return len(self.left_test)