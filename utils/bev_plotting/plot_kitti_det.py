import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import plot_bird_view

DETECTION_PATH = '/home/robesafe/Miguel/3d-detection-pipeline/results/geometric_estimation/'
GT_PATH = '/media/robesafe/SSD_SATA/KITTI_DATASET/label_2/'

def main():

    assert os.path.isdir(DETECTION_PATH)
    names_list = [x.strip('.txt') for x in sorted(os.listdir(DETECTION_PATH)) if x.endswith('.txt')]
    # det_list = [os.path.join(DETECTION_PATH,x) for x in sorted(os.listdir(DETECTION_PATH)) if x.endswith('.txt')]
    # gt_list = [os.path.join(GT_PATH,x) for x in sorted(os.listdir(GT_PATH)) if x.endswith('.txt')]

    if not os.path.isdir(os.path.join(DETECTION_PATH,'bev_plotting')):
        os.mkdir(os.path.join(DETECTION_PATH,'bev_plotting'))

    for name in tqdm(names_list):

        annotation_header = ['type','truncated','occluded','alpha','left','top','right','bottom','h','w','l','x','y','z','rot_y','score']
        det_fd = pd.read_csv(os.path.join(DETECTION_PATH,name+'.txt'),names=annotation_header, sep=' ')
        gt_fd = pd.read_csv(os.path.join(GT_PATH,name+'.txt'),names=annotation_header, sep=' ')

        # Define bev limits
        x_max = 30
        z_max = 50
        pix_x = 1000
        pix_z = 2000
        im = np.zeros((pix_z,pix_x,3))
        cv2.circle(im, (int(pix_x/2),int(pix_z)), 4, (255,0,0), thickness=5)

        for idx,row in gt_fd.iterrows():
        
            plot_bird_view(im,(row['h'],row['w'],row['l']),row['alpha'],row['rot_y']-row['alpha'],(row['x'],row['y'],row['z']),(0,255,0),x_max,z_max)

        for idx,row in det_fd.iterrows():
            
            plot_bird_view(im,(row['h'],row['w'],row['l']),row['alpha'],row['rot_y']-row['alpha'],(row['x'],row['y'],row['z']),(255,0,0),x_max,z_max)

        cv2.imwrite(os.path.join(DETECTION_PATH,'bev_plotting',name+'.png'),im)

if __name__ == '__main__':
    main()
