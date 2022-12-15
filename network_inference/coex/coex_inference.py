
import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = True

from stereo import Stereo
from ruamel.yaml import YAML
from tqdm import tqdm



DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/coex')
# IMAGE_LIST = os.path.join(ROOT_PATH,'imagenes.txt') 
IMAGE_LIST = os.path.join(ROOT_PATH,'ImageSets/val.txt') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/coex/last.ckpt') 
KITTI_STEREO_BASELINE = 0.54    # distance between left and right images in meters (KITTI)

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

    with open(IMAGE_LIST,'r') as f_content:
        images_names = list(map(lambda x: x.split('\n')[0], f_content.readlines()))
    
    cfg = load_configs(os.path.join(ROOT_PATH,'network_inference/coex/configs/stereo/cfg_coex.yaml'))
    cfg['stereo_ckpt'] = WEIGHTS

    pose_ssstereo = Stereo.load_from_checkpoint(cfg['stereo_ckpt'],
                                                strict=False,
                                                cfg=cfg).cuda()

    pose_ssstereo.eval()

    acc_fps = 0

    for image_n in tqdm(images_names):
        
        imgL,imgR = load_images(DATAPATH,image_n)
        calib = load_calib(DATAPATH,image_n)
        fy = float(calib[2].strip("P2: ").split(' ')[5])

        
        
        starter.record()
        
        with torch.no_grad():
            half_precision = True
            with torch.cuda.amp.autocast(enabled=half_precision):
                
                img = torch.cat([imgL, imgR], 0)
                disp = pose_ssstereo(img, training=False)
                disp = torch.squeeze(disp, 0)
        
        ender.record()
        torch.cuda.synchronize()

        # print(f'FPS: {1/(starter.elapsed_time(ender)/1000)}')
        acc_fps += 1/(starter.elapsed_time(ender)/1000)
        
        predicted_depth = (fy*KITTI_STEREO_BASELINE)/disp.cpu().numpy()
        
        np.save(SAVE_PATH + '/depth_maps/' + image_n , predicted_depth)

    print(f'Average FPS: {acc_fps/len(images_names)}')





if __name__ == '__main__':

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if not os.path.isdir(SAVE_PATH + '/depth_maps/'):
        os.mkdir(SAVE_PATH + '/depth_maps/')

    main()