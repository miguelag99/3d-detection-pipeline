
import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from stereo import Stereo
from ruamel.yaml import YAML
from tqdm import tqdm

import dataloader.SHIFT_loader  as SHIFT_loader

DATAPATH = '/home/robesafe/Datasets/shift_dataset/training'
ROOT_PATH = '/home/robesafe/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/coex_shift_w_kitti_weights')
# IMAGE_LIST = os.path.join(ROOT_PATH,'imagenes.txt') 
IMAGE_LIST = os.path.join('/home/robesafe/Datasets/shift_dataset','ImageSets','val.txt') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/coex/kitti_last.ckpt') 
STEREO_BASELINE = 0.2    # distance between left and right images in meters

im_dim = (cv2.imread(DATAPATH + '/image_2/000000.png')).shape
print(f'Image shape: {im_dim}')



__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg

def load_images(datset_path, name):
    # TODO: change to dataloader
    image_l_path = cv2.imread(os.path.join(DATAPATH,'image_2',name+'.png'))
    image_r_path = cv2.imread(os.path.join(DATAPATH,'image_3',name+'.png'))

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std']),
    ])


    return transf(image_l_path).unsqueeze(0).cuda(), transf(image_r_path).unsqueeze(0).cuda()

def load_calib(dataset_path, name):
    # TODO: change to dataloader

    with open(os.path.join(DATAPATH,'calib',f'{name}.txt'),"r") as fd:
        calib_file = list(map(lambda x: x.split('\n')[0], fd.readlines()))

   
    return calib_file


def main():

    # LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Dataloader
    TestImgLoader = torch.utils.data.DataLoader(
        SHIFT_loader.SHIFT_Dataset(DATAPATH, IMAGE_LIST, False),
        batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    with open(IMAGE_LIST,'r') as f_content:
        images_names = list(map(lambda x: x.split('\n')[0], f_content.readlines()))
    
    cfg = load_configs(os.path.join(ROOT_PATH,'network_inference/coex/configs/stereo/cfg_coex.yaml'))
    cfg['stereo_ckpt'] = WEIGHTS

    pose_ssstereo = Stereo.load_from_checkpoint(cfg['stereo_ckpt'],
                                                strict=False,
                                                cfg=cfg).cuda()

    pose_ssstereo.eval()

    acc_fps = 0
    
    tqdm_eval_loader = tqdm(TestImgLoader, total=len(TestImgLoader),desc="Inference")
    for batch_idx, (imgL_crops, imgR_crops, calib, H, W, filename) in enumerate(tqdm_eval_loader):
        output = []

        with torch.no_grad():
            half_precision = True
            with torch.cuda.amp.autocast(enabled=half_precision):
                for imgL_crop, imgR_crop in zip(imgL_crops, imgR_crops):
                    imgL = imgL_crop.cuda()
                    imgR = imgR_crop.cuda()
                    img = torch.cat([imgL, imgR], 0)
                    disp = pose_ssstereo(img, training=False)
                    disp = torch.squeeze(disp, 0)
                    output.append(disp)
        pred_disp = torch.cat(output, dim=0)


        if pred_disp.shape[1] != im_dim[1]:
            # Pad with zeros to match the original image size
            pred_disp = F.pad(pred_disp, (0, im_dim[1] - pred_disp.shape[1], 0, im_dim[0]-pred_disp.shape[0]), 
                              mode='constant', value=0)

        predicted_depth = calib/pred_disp.data.cpu().numpy()
        
        np.save(SAVE_PATH + '/depth_maps/' + filename[0], predicted_depth)



    # for image_n in tqdm(images_names):
        
    #     imgL,imgR = load_images(DATAPATH,image_n)

    #     B, C, H, W = imgL.shape

    #     print(f'Image shape: {imgL.shape}')

    #     top_pad = 384 - H
    #     right_pad = 1248 - W
    #     imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
    #     imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
        
    #     print(f'Image shape after padding: {imgL.shape}')

    #     calib = load_calib(DATAPATH,image_n)
    #     fy = float(calib[2].strip("P2: ").split(' ')[5])

        
        
    #     starter.record()
        
    #     with torch.no_grad():
    #         half_precision = True
    #         with torch.cuda.amp.autocast(enabled=half_precision):
                
    #             img = torch.cat([imgL, imgR], 0)
    #             disp = pose_ssstereo(img, training=False)
    #             disp = torch.squeeze(disp, 0)
        
    #     ender.record()
    #     torch.cuda.synchronize()

        
    #     # print(f'FPS: {1/(starter.elapsed_time(ender)/1000)}')

    #     if int(image_n) != int(images_names[0]):
    #         acc_fps += 1/(starter.elapsed_time(ender)/1000)
        
    #     predicted_depth = (fy*KITTI_STEREO_BASELINE)/disp.cpu().numpy()

       
    #     np.save(SAVE_PATH + '/depth_maps/' + image_n , predicted_depth[0:H,0:W])

    # print(f'Average FPS: {acc_fps/(len(images_names)-1)}')





if __name__ == '__main__':

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if not os.path.isdir(SAVE_PATH + '/depth_maps/'):
        os.mkdir(SAVE_PATH + '/depth_maps/')

    main()