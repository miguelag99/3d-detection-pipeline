import numpy as np
import os
import math
import pandas as pd
import argparse

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

color_map = {
    'Car':  (0,0,142) ,
    'Pedestrian': (220, 20, 60),
    'Cyclist': (119, 11, 32),
    'Truck': (0,0,70),
    'Van': (0,0,90),
    'Misc': (81,0,81),
    'Tram': (0,0,230),
}

def main(args):

    if args.dataset == 'kitti':
        DATASET_PATH = "/media/robesafe/SSD_SATA/KITTI_DATASET/training/"
        IM_SIZE = (1242,375)
    else:
        DATASET_PATH = "/media/robesafe/SSD_SATA/shift_dataset/training"
        IM_SIZE = (1280,800)
    
    VELO_PATH = os.path.join(DATASET_PATH,"velodyne/")
    CALIB_PATH = os.path.join(DATASET_PATH,"calib/")
    GT_PATH = os.path.join(DATASET_PATH,'label_2/')

    id  = args.id

    if args.img_list != '':
        ids = np.loadtxt(args.img_list,dtype=np.int16)
    else:
        ids = [id]


    for n in tqdm(ids):

        assert os.path.isfile(os.path.join(GT_PATH, '{:06d}.txt'.format(n))), "File not found"

        im_name = '{:06d}'.format(n)
        gt_file = os.path.join(GT_PATH, im_name + '.txt')
        calib_file = os.path.join(CALIB_PATH, im_name + '.txt')
        velo_file = os.path.join(VELO_PATH, im_name + '.bin')

        # Transformation matrices
        calib_file = open(calib_file,"r").readlines()

        p2 = calib_file[2].strip('\n').strip("P2: ").split(' ')
        p2 = np.matrix([float(x) for x in p2]).reshape(3,4)
        p2 = np.vstack((p2,np.array((0,0,0,1))))
        
        R0_rect = calib_file[4].strip('\n').strip('R0_rect: ').split(' ')
        R0_rect = np.matrix([float(x) for x in R0_rect]).reshape(3,3)
        R0_rect = np.hstack((R0_rect,np.array(([0],[0],[0]))))
        R0_rect = np.vstack((R0_rect,np.array((0,0,0,1))))

        Tr_velo_to_cam = calib_file[5].strip('\n').strip('Tr_velo_to_cam: ').split(' ')
        Tr_velo_to_cam = np.matrix([float(x) for x in Tr_velo_to_cam]).reshape(3,4)
        Tr_velo_to_cam = np.vstack((Tr_velo_to_cam,np.array([0,0,0,1])))

        Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
        R0_rect_inv = np.linalg.inv(R0_rect)

        # Filter points outside image FOV
        # velo = np.fromfile(velo_file,dtype=np.float32, count=-1).reshape([-1,4])
        # velo[:,3] = 1
        # proj_pcl = (p2 @ R0_rect @ Tr_velo_to_cam @ velo.T).T
        # proj_pcl[:,0] = proj_pcl[:,0] / proj_pcl[:,2]
        # proj_pcl[:,1] = proj_pcl[:,1] / proj_pcl[:,2]
        # filter_pcl = np.where((proj_pcl[:,0] < 0) | (proj_pcl[:,0] > IM_SIZE[0]) |
        #                       (proj_pcl[:,1] < 0) | (proj_pcl[:,1] > IM_SIZE[1]))
        # velo = np.delete(velo,filter_pcl,axis=0)
        # velo = np.delete(velo,(velo[:,0]<0)|(velo[:,0]>50),axis=0)

        # Prepare visualizator
        ego_pose = (750,1000)
        pix_2_m_ratio = args.ratio
        img = prepare_visualizator(pix_2_m_ratio = pix_2_m_ratio,ego_pose = ego_pose)

        # Print obstacles
        try:
            f_kitti = pd.read_csv(gt_file,sep=" ",header=None)
            f_kitti.columns= ["Class","truncated","occluded","alpha","left","top","rigth","bottom","h","w","l","x","y","z","rot"]
            df_filter=f_kitti['Class']!='DontCare'
            f_kitti = f_kitti[df_filter]

            for i,row in f_kitti.iterrows(): 
                # Plot 3D boxes
                if row['Class'] in color_map.keys():
                    box_3d = create_3d_bbox(row['rot'],(row['h'],row['w'],row['l']),(row['x'],row['y'],row['z']))[:,4:]
                    plot_3d_bev_box(img,box_3d,ego_pose,pix_2_m_ratio,color_map[row['Class']])
        except:
            print("No ground truth found at {}".format(gt_file))

        if args.detect_path != '':
            detection_txt = os.path.join(args.detect_path,im_name+'.txt')
            try:
                f_detect = pd.read_csv(detection_txt,sep=" ",header=None)
                f_detect.columns= ["Class","truncated","occluded","alpha","left","top","rigth","bottom","h","w","l","x","y","z","rot","score"]
                
                for i,row in f_detect.iterrows():
                    # Plot 3D boxes
                    if row['Class'] in color_map.keys():
                        box_3d = create_3d_bbox(row['rot'],(row['h'],row['w'],row['l']),(row['x'],row['y'],row['z']))[:,4:]
                        plot_3d_bev_box(img,box_3d,ego_pose,pix_2_m_ratio,color_map[row['Class']],2)
            except:
                print("No detections found at {}".format(detection_txt))

        if args.save_path != '':
            cv2.imwrite(os.path.join(args.save_path,im_name+'_bev.png'),img)
        else:
            cv2.imshow('Color image', img.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def prepare_visualizator(height = 1000, width = 1500, pix_2_m_ratio = 10,
                         min_dist = 10, max_dist = 70, step = 10, fov = 90,
                         ego_pose = (750,900)):
    
    img = np.zeros((height,width,3))

    circ_m = np.arange(min_dist,max_dist+step,step)
    circ_text = [str(x) for x in circ_m]

    r = 255
    g = 255
    b = 255

    # Print distance circles
    for d in reversed(circ_m):
        r *= 0.75
        g *= 0.75
        b *= 0.75
        cv2.circle(img,ego_pose, int(d*pix_2_m_ratio), (int(b),int(g),int(r)), -1)

    # Print FOV lines
    fov = 90*np.pi/180
    cv2.line(img,ego_pose,
             (int(ego_pose[0]+ math.sin(fov/2)*max_dist*pix_2_m_ratio),int(ego_pose[1]- math.cos(fov/2)*max_dist*pix_2_m_ratio)),
             (255,0,255),2)
    cv2.line(img,ego_pose,
            (int(ego_pose[0]- math.sin(fov/2)*max_dist*pix_2_m_ratio),int(ego_pose[1]- math.cos(fov/2)*max_dist*pix_2_m_ratio)),
            (255,0,255),2)

    # Print distance text
    for d in reversed(circ_m):
        pix_u = int(ego_pose[0]+ math.sin(fov/2)*d*pix_2_m_ratio) + int(2*pix_2_m_ratio)
        pix_v = int(ego_pose[1]- math.cos(fov/2)*d*pix_2_m_ratio) + int(2*pix_2_m_ratio)
        cv2.putText(img, circ_text.pop(), (pix_u,pix_v),
                    cv2.FONT_HERSHEY_SIMPLEX,0.1*pix_2_m_ratio, (255,0,255), 2, cv2.LINE_AA)
        
    return img

def plot_3d_bev_box(img,box_3d,ego_pose,pix_2_m_ratio,color=(0,255,0),thickness=-1):

        # for pt in box_3d.T:       
        #     x,y,z = pt[0],pt[1],pt[2]
        #     u = int(x*pix_2_m_ratio + ego_pose[0])
        #     v = int(z*pix_2_m_ratio + ego_pose[1] - 2*z*pix_2_m_ratio)
        #     cv2.circle(img,(u,v), 5, (0,0,255), -1)


        u1,v1 = int(box_3d[0,0]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,0]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,0]*pix_2_m_ratio)
        u2,v2 = int(box_3d[0,1]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,1]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,1]*pix_2_m_ratio)
        # cv2.line(img,(u1,v1),(u2,v2),color,2,cv2.LINE_AA)
        p1 = (u1,v1)
        p2 = (u2,v2)

        u1,v1 = int(box_3d[0,1]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,1]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,1]*pix_2_m_ratio)
        u2,v2 = int(box_3d[0,2]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,2]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,2]*pix_2_m_ratio)
        # cv2.line(img,(u1,v1),(u2,v2),color,2,cv2.LINE_AA)

        p3 = (u2,v2)

        u1,v1 = int(box_3d[0,2]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,2]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,2]*pix_2_m_ratio)
        u2,v2 = int(box_3d[0,3]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,3]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,3]*pix_2_m_ratio)
        # cv2.line(img,(u1,v1),(u2,v2),color,2,cv2.LINE_AA)

        p4 = (u2,v2)

        u1,v1 = int(box_3d[0,3]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,3]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,3]*pix_2_m_ratio)
        u2,v2 = int(box_3d[0,0]*pix_2_m_ratio + ego_pose[0]),int(box_3d[2,0]*pix_2_m_ratio + ego_pose[1] - 2*box_3d[2,0]*pix_2_m_ratio)
        # cv2.line(img,(u1,v1),(u2,v2),color,2,cv2.LINE_AA)

        cv2.drawContours(img, [np.array([p1,p2,p3,p4])], -1, color, thickness, cv2.LINE_AA)


def rotation_mat(angle):

    # Creates a rotation matrix around the height axis (y in camera and z in lidar)
    sin = math.sin(angle-np.pi/2)
    cos = math.cos(angle-np.pi/2)
    # rot = np.vstack(([cos,-sin,0],[sin,cos,0],[0,0,1]))
    rot = np.vstack(([cos,0,sin],[0,1,0],[-sin,0,cos]))
    # rot = np.vstack(([1,0,0],[0,cos,-sin],[0,sin,cos]))
  
    return rot


def create_3d_bbox(rot,dim,loc):

    rot = rotation_mat(rot)
        
    (h,w,l) = (dim[0],dim[1],dim[2])
    (x,y,z) = (loc[0],loc[1],loc[2])
    
    cube = np.vstack(([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2],[-h,-h,-h,-h,0,0,0,0],[-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2]))
    offset = np.vstack((np.full((1,8),x),np.full((1,8),y),np.full((1,8),z)))  
        
    box = np.dot(rot,cube) + offset
    box = np.vstack((box,np.full((1,8),1)))

    return box


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot KITTI and SHIFT dataset ground truth')
    # Add argparser arguments
    parser.add_argument('-i', '--id', type=int, help='Image id', default=0)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name', default='kitti',choices=['kitti','shift'])
    parser.add_argument('-r','--ratio', type=float, help='Pixel to meter ratio', default=10)
    parser.add_argument('-det_p','--detect_path',type=str,help='Detection path',default='')
    parser.add_argument('-s','--save_path',type=str,help='Save path',default='')
    parser.add_argument('-im_l','--img_list',type=str,help='Image list',default='')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)

