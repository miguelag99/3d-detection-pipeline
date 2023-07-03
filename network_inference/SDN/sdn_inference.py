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

DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/SDN_kitti_w_shift_weights')
IMAGE_LIST = os.path.join(ROOT_PATH,'ImageSets','val.txt') 


# WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/SDN/fine_tune_kitti_trainval_last.tar') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/SDN/shift_checkpoint_50.pth.tar')
DATA_TAG = 'pruebas'


im_dim = (cv2.imread(DATAPATH + '/image_2/000000.png')).shape
print(im_dim)

def main():

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    TrainImgLoader = None
    # Data Loader for KITTI
    import dataloader.KITTI_submission_loader  as KITTI_submission_loader
    TestImgLoader = torch.utils.data.DataLoader(
        KITTI_submission_loader.SubmiteDataset(DATAPATH, IMAGE_LIST, False),
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
    for batch_idx, (imgL_crop, imgR_crop, calib, H, W, filename) in enumerate(tqdm_eval_loader):
        # print(f'imgL shape: {imgL_crop.shape} {type(imgL_crop)}, imgR shape: {imgR_crop.shape} {type(imgR_crop)}')
        pred_disp = inference(imgL_crop, imgR_crop, calib, model, H, W)
        
        for idx, name in enumerate(filename):
            # np.save(SAVE_PATH + '/depth_maps/' + DATA_TAG + '/' + name, pred_disp[idx][-H[idx]:, :W[idx]])
            np.save(SAVE_PATH + '/depth_maps/' + name, pred_disp[idx])

    import sys
    sys.exit()




    
def inference(imgL, imgR, calib, model, H, W):

    # LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    model.eval()
    
    imgL, imgR, calib = imgL.cuda(), imgR.cuda(), calib.float().cuda()
    # print(f'imgL shape: {imgL.shape} {type(imgL)}, imgR shape: {imgR.shape} {type(imgR)}, calib shape: {calib.shape} {type(calib)}')

    start_time = time.time()
    with torch.no_grad():  
        starter.record()      
        output = model(imgL, imgR, calib)
        ender.record()
        torch.cuda.synchronize()

    # print("CUDA elapsed time: %s",str(starter.elapsed_time(ender)))
    
    elapsed_time = time.time() - start_time
    # print('Sys elapesed time: %s' % str(elapsed_time))
    
    # print(f"Out batch size: {output.shape}")

    if args.data_type == 'disparity':
        output = disp2depth(output, calib)

    pred_disp = output.data.cpu().numpy()


   
    # disp_as_img = cv2.resize(pred_disp[0,:,:], (im_dim[1], im_dim[0]), interpolation=cv2.INTER_NEAREST)
    # pred_disp = np.zeros((1, im_dim[0], im_dim[1]))
    # pred_disp[0,:,:] = disp_as_img


    return pred_disp[:, :H.data.cpu(), :W.data.cpu()]

def disp2depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth

if __name__ == '__main__':
    main()