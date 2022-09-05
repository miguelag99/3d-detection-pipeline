import configargparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

import models
from dataloader import KITTILoader3D
from dataloader import KITTILoader_dataset3d


parser = configargparse.ArgParser(description='PSMNet')
parser.add_argument('--save_path', type=str, default='',
                    help='path to save the log, tensorbaord and checkpoint')
# network
parser.add_argument('--data_type', default='depth', choices=['disparity', 'depth'],
                    help='the network can predict either disparity or depth')
parser.add_argument('--arch', default='SDNet', choices=['SDNet', 'PSMNet'],
                    help='Model Name, default: SDNet.')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity, the range of the disparity cost volume: [0, maxdisp-1]')
parser.add_argument('--down', type=float, default=2,
                    help='reduce x times resolution when build the depth cost volume')
parser.add_argument('--maxdepth', type=int, default=80,
                    help='the range of the depth cost volume: [1, maxdepth]')

# dataset
parser.add_argument('--kitti2015', action='store_true',
                    help='If false, use 3d kitti dataset. If true, use kitti stereo 2015, default: False')
parser.add_argument('--dataset', default='kitti', choices=['sceneflow', 'kitti'],
                    help='train with sceneflow or kitti')
parser.add_argument('--datapath', default='',
                    help='root folder of the dataset')
parser.add_argument('--split_train', default='Kitti/object/train.txt',
                    help='data splitting file for training')
parser.add_argument('--split_val', default='Kitti/object/subval.txt',
                    help='data splitting file for validation')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of training epochs')
parser.add_argument('--btrain', type=int, default=12,
                    help='training batch size')
# parser.add_argument('--bval', type=int, default=4,
#                     help='validation batch size')
parser.add_argument('--bval', type=int, default=1,
                    help='validation batch size')
parser.add_argument('--workers', type=int, default=8,
                    help='number of dataset workers')
args = parser.parse_args()

DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
SAVE_PATH = './results/'
IMAGE_LIST = './imagenes.txt'
WEIGHTS = './model_best.pth.tar'
DATA_TAG = 'pruebas'

def main():

    # Data Loader
    TrainImgLoader = None
    import dataloader.KITTI_submission_loader  as KITTI_submission_loader
    TestImgLoader = torch.utils.data.DataLoader(
        KITTI_submission_loader.SubmiteDataset(DATAPATH, IMAGE_LIST, False),
        batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)

    # Load Model
    model = models.__dict__[args.arch](maxdepth=args.maxdepth, maxdisp=args.maxdisp, down=args.down)

    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True



    if os.path.isfile(WEIGHTS):
            print("=> loading checkpoint '{}'".format(WEIGHTS))
            checkpoint = torch.load(WEIGHTS)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                     .format(WEIGHTS, checkpoint['epoch']))
    else:
        print('[Attention]: Can not find checkpoint {}'.format(WEIGHTS))
    

    os.makedirs(SAVE_PATH + '/depth_maps/' + DATA_TAG, exist_ok=True)

    tqdm_eval_loader = tqdm(TestImgLoader, total=len(TestImgLoader))
    for batch_idx, (imgL_crop, imgR_crop, calib, H, W, filename) in enumerate(tqdm_eval_loader):
        
        pred_disp = inference(imgL_crop, imgR_crop, calib, model)
        
        for idx, name in enumerate(filename):
            np.save(SAVE_PATH + '/depth_maps/' + DATA_TAG + '/' + name, pred_disp[idx][-H[idx]:, :W[idx]])
    import sys
    sys.exit()




    
def inference(imgL, imgR, calib, model):
    model.eval()
    imgL, imgR, calib = imgL.cuda(), imgR.cuda(), calib.float().cuda()
    torch.cuda.synchronize()
    with torch.no_grad():
        tic = time.time()
        output = model(imgL, imgR, calib)
        print(time.time()-tic)
    if args.data_type == 'disparity':
        output = disp2depth(output, calib)
    pred_disp = output.data.cpu().numpy()

    return pred_disp

def disp2depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth

if __name__ == '__main__':
    main()