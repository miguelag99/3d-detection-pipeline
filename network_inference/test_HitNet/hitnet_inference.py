import os
import time

import numpy as np
import torch
from tqdm import tqdm
import PIL.Image as Image
import cv2

from utils import HitNet, ModelType, draw_disparity, draw_depth, load_img

DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/Miguel/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/hitnet')
IMAGE_LIST = os.path.join(ROOT_PATH,'imagenes.txt') 
# IMAGE_LIST = os.path.join(ROOT_PATH,'ImageSets/val.txt') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/Hitnet/eth3d.pb') 

KITTI_STEREO_BASELINE = 0.54    # distance between left and right images in meters (KITTI)



def main():

    # Select model type
    # model_type = ModelType.middlebury
    # model_type = ModelType.flyingthings
    model_type = ModelType.eth3d


    # Load model
    hitnet_depth = HitNet(WEIGHTS, model_type)   
    # hitnet_depth.plot_model('aaaa.png')

    ########### Tensorflow dataloader????

    left_fold = os.path.join(DATAPATH,'image_2/')
    right_fold = os.path.join(DATAPATH,'image_3/')
    calib_fold = os.path.join(DATAPATH,'calib/')
    with open(IMAGE_LIST, 'r') as f:
        image = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    image = sorted(image)

    left_images = [left_fold + img + '.png' for img in image]
    right_images = [right_fold + img + '.png' for img in image]
    # calib_images = [calib_fold + img + '.txt' for img in image]

    elapsed_time = 0

    for batch_idx in tqdm(range(len(image))):
        imgL = np.asarray(Image.open(left_images[batch_idx]).convert('RGB'))[:,:1000]
        imgR = np.asarray(Image.open(right_images[batch_idx]).convert('RGB'))[:,:1000]

        print(f'{type(imgL)} de tamaño {imgL.shape}')


        # LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # warm up
        disparity_map = hitnet_depth(imgL, imgR)

        # Estimate the depth

        starter.record() 
        disparity_map = hitnet_depth(imgL, imgR)
        ender.record()
        torch.cuda.synchronize()

        elapsed_time = elapsed_time + starter.elapsed_time(ender)


        # color_disparity = draw_disparity(disparity_map)
        # cobined_image = np.hstack((imgL, imgR, color_disparity))

        # cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
        # cv2.imshow("Estimated disparity", cobined_image)
        # cv2.waitKey(0)


    print("Mean CUDA elapsed time: ",str(elapsed_time/len(image)))
    # print("CPU elapsed time: ",str(t2-t1))

if __name__ == '__main__':
    main()