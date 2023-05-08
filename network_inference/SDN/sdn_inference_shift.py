import configargparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import numpy as np
import time

from tqdm import tqdm

import models


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
parser.add_argument('--workers', type=int, default=0,
                    help='number of dataset workers')
args = parser.parse_args()


DATAPATH = '/home/robesafe/Datasets/shift_dataset/training'
ROOT_PATH = '/home/robesafe/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/SDN_shift')
IMAGE_LIST = '/home/robesafe/Datasets/shift_dataset/trainval.txt'

WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/SDN/shift_checkpoint_25.pth.tar') 
DATA_TAG = 'pruebas'

im_dim = (cv2.imread(DATAPATH + '/image_2/000000.png')).shape

def main():

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    TrainImgLoader = None
    # Data Loader for shift
    import dataloader.SHIFT_loader  as SHIFT_loader
    TestImgLoader = torch.utils.data.DataLoader(
        SHIFT_loader.SHIFT_Dataset(DATAPATH, IMAGE_LIST, False),
        batch_size=args.bval, shuffle=False, num_workers=0, drop_last=False)

        


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

    # Inference and save with time measurements
    if not os.path.isdir(SAVE_PATH + '/depth_maps/'):
        os.makedirs(SAVE_PATH + '/depth_maps/', exist_ok=True)

    tqdm_eval_loader = tqdm(TestImgLoader, total=len(TestImgLoader),desc="Inference")
    for batch_idx, (imgL_crops, imgR_crops, calib, H, W, filename) in enumerate(tqdm_eval_loader):
        # print(f'imgL shape: {imgL_crop[0].shape} {type(imgL_crop)}, imgR shape: {imgR_crop[0].shape} {type(imgR_crop)}')
        
        pred_disp = inference(imgL_crops, imgR_crops, calib, model)
        
        for idx, name in enumerate(filename):
            # np.save(SAVE_PATH + '/depth_maps/' + DATA_TAG + '/' + name, pred_disp[idx][-H[idx]:, :W[idx]])
            np.save(SAVE_PATH + '/depth_maps/' + name, pred_disp[idx])

    import sys
    sys.exit()




    
def inference(imgL_crops, imgR_crops, calib, model):

    model.eval()
    
    calib = calib.float().cuda()

    # print(f'imgL shape: {imgL.shape} {type(imgL)}, imgR shape: {imgR.shape} {type(imgR)}, calib shape: {calib.shape} {type(calib)}')
    
    output = []
    with torch.no_grad():  
        for imgL_crop, imgR_crop in zip(imgL_crops, imgR_crops):
            imgL = imgL_crop.cuda()
            imgR = imgR_crop.cuda()
            output.append(model(imgL, imgR, calib))

    
    
    if args.data_type == 'disparity':
        for i in range(len(output)):
            output[i] = disp2depth(output[i], calib)
    pred_disp = torch.concat(output, dim=1)

    # print(f'pred_disp shape: {pred_disp.shape} {type(pred_disp)}')
    # print(im_dim[1] - pred_disp.shape[2], im_dim[0]-pred_disp.shape[1])

    if pred_disp.shape[1] != im_dim[1]:
            # Pad with zeros to match the original image size
            pred_disp = F.pad(pred_disp, (0, im_dim[1] - pred_disp.shape[2], 0, im_dim[0]-pred_disp.shape[1]), mode='constant', value=0)

    # print(f'final shape: {pred_disp.shape} {type(pred_disp)}')

    # output = [o.data.cpu().numpy() for o in output]
    # pred_disp =  np.hstack(output)

    

    
    # disp_as_img = cv2.resize(pred_disp[0,:,:], (im_dim[1], im_dim[0]))
    # pred_disp = np.zeros((1, im_dim[0], im_dim[1]))
    # pred_disp[0,:,:] = disp_as_img

    return pred_disp.data.cpu().numpy()

def disp2depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth

if __name__ == '__main__':
    main()