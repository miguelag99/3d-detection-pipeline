import os

import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/Miguel/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/geometric_estimation')
IMAGE_LIST_TXT = os.path.join(ROOT_PATH,'ImageSets/val.txt')
YOLO_REPO = os.path.join(ROOT_PATH,'network_inference/geometric_3d_estimation/yolov5')
WEIGHTS = os.path.join(YOLO_REPO,'yolov5l.pt')

CLASS_DICT = {
    "Pedestrian": 0,
    "Cyclist": 1,
    "Car": 2
}

CLASS_MEAN_3D = {
    "Pedestrian": [1.761, 0.660, 0.842],
    "Cyclist": [1.737, 0.597, 1.763],
    "Car": [1.526, 1.628, 3.884]
}

CAMERA_HEIGHT = 1.65

class Yolo_inference():
    def __init__(self,weights):
        self.yolo = torch.hub.load(YOLO_REPO, 'custom', path=weights, source='local')
        self.yolo.conf = 0.5
        self.yolo.iou = 0.45
        # object idx: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'traffic light', 'stop sign']
        # OBJECT_DETECTOR_2D_list = [0,1,2,3,5,9,11]
        self.yolo.classes = [0,1,2]
        self.yolo.agnostic = False

    def detection2d_pipeline(self, img_batch):
        detections = self.yolo(img_batch).pandas().xyxy
        return detections


def read_params(file_path):

    camera_param = open(file_path,'r')
    matrix = camera_param.readlines()
    camera_param.close()
    matrix = (matrix[2].split(':'))[1].split(' ')
    matrix.pop(0)
    matrix[11] = matrix[11].rstrip('\n')
    matrix = [float(i) for i in matrix]
    
    p = np.vstack((matrix[0:4],matrix[4:8],matrix[8:12]))
    return p


def pix2realworld(P,detections,result_name, im):
    P_inv = np.linalg.pinv(P)
    result_fd = open(result_name,'w')
 
    detections = detections.round({'xmin':0, 'ymin':0, 'xmax':0, 'ymax':0})
    detections = detections.astype({'xmin':'int', 'ymin':'int', 'xmax':'int', 'ymax':'int'})

    for i,obj in detections.iterrows():

        cls = list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(obj['class'])]

        centroid_x = (obj['xmin']+obj['xmax'])/2 
        pixels = np.array([[centroid_x],[obj['ymax']],[1]])

        cv2.rectangle(im,(obj['xmin'],obj['ymin']),(obj['xmax'],obj['ymax']),(255,0,0),2)

        p_camera = np.dot(P_inv,pixels)

        if (p_camera[1] != 0):
            K = CAMERA_HEIGHT / p_camera[1]
        else:
            return None
        pose_est = np.dot(p_camera, K)
        
        alpha = 0
        theta_ray = 0
        bbox = (obj['xmin'],obj['ymin'],obj['xmax'],obj['ymax'])
        conf = obj['confidence']
        dim = CLASS_MEAN_3D[cls]

        result_fd.write("{} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(\
        cls,alpha,bbox[0],bbox[1],bbox[2],bbox[3],\
        dim[0],dim[1],dim[2],pose_est[0],pose_est[1],pose_est[2],alpha+theta_ray,conf)+os.linesep)

        # cv2.imshow('Color image', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    result_fd.close()

def main():

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    image_fd = open(IMAGE_LIST_TXT,'r')
    im_names = image_fd.readlines()
    image_fd.close()
    image_list = [os.path.join(DATAPATH+'/image_2',name.strip('\n')+'.png') for name in im_names]
    calib_list = [os.path.join(DATAPATH+'/calib',name.strip('\n')+'.txt') for name in im_names]
    result_list = [os.path.join(SAVE_PATH,name.strip('\n')+'.txt') for name in im_names]


    Yolo_object = Yolo_inference(WEIGHTS)
    
    image_iter = tqdm(image_list,total=len(image_list))

    for (i,im_name) in enumerate(image_iter):

        # im_name = image_iter[i]
        calib_name = calib_list[i]
        result_name = result_list[i]

        input_image = cv2.imread(im_name)
        calib = read_params(calib_name)

        predicted_obj = Yolo_object.detection2d_pipeline(input_image)
        
        if predicted_obj:
            pix2realworld(calib,predicted_obj[0], result_name, input_image) # Save detections
        else:
            open(result_name, 'w').close() # Create empty detections file






if __name__ == '__main__':
    main()