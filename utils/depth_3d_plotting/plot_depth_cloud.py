
from turtle import bgcolor
import numpy as np
import os
import mayavi.mlab
import pandas as pd
import math

from generate_lidar_from_depth import depth_2_pointcloud

ROOT_PATH = '/home/robesafe/Miguel/3d-detection-pipeline'
DEPTH_PATH = os.path.join(ROOT_PATH,'results/mobilestereonet/depth_maps/')
KITTI_PATH = "/home/robesafe/Datasets/kitti_pseudolidar/training/"
VELO_PATH = os.path.join(KITTI_PATH,"velodyne/")
CALIB_PATH = os.path.join(KITTI_PATH,"calib/")
GT_PATH = os.path.join(KITTI_PATH,'label_2/')

def main(transform_coord = False):

    # Load and transform data
    id = 0

    depth_files = [DEPTH_PATH + name for name in sorted(os.listdir(DEPTH_PATH)) if name.endswith('.npy')]
    gt_file = os.path.join(GT_PATH, depth_files[id].split('/')[-1].split('.')[0] + '.txt')
    calib_file = os.path.join(CALIB_PATH, depth_files[id].split('/')[-1].split('.')[0] + '.txt')
    velo_file = os.path.join(VELO_PATH, depth_files[id].split('/')[-1].split('.')[0] + '.bin')
    
    pointcloud = depth_2_pointcloud(calib_file,depth_dir=depth_files[id],save_dir=None,max_high=1).reshape([-1,4])
    velo = np.fromfile(velo_file,dtype=np.float32, count=-1).reshape([-1,4])
    
    print(f'Shape velo:{velo.shape} and shape depth:{pointcloud.shape}')
    # print(velo)
    # print(pointcloud)


    # Transformation matrices

    calib_file = open(calib_file,"r").readlines()
    
    p2 = calib_file[2].strip('\n').strip("P2: ").split(' ')
    p2 = np.matrix([float(x) for x in p2]).reshape(3,4)
    
    R0_rect = calib_file[4].strip('\n').strip('R0_rect: ').split(' ')
    R0_rect = np.matrix([float(x) for x in R0_rect]).reshape(3,3)
    R0_rect = np.hstack((R0_rect,np.array(([0],[0],[0]))))
    R0_rect = np.vstack((R0_rect,np.array((0,0,0,1))))

    Tr_velo_to_cam = calib_file[5].strip('\n').strip('Tr_velo_to_cam: ').split(' ')
    Tr_velo_to_cam = np.matrix([float(x) for x in Tr_velo_to_cam]).reshape(3,4)
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam,np.array([0,0,0,1])))

    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    R0_rect_inv = np.linalg.inv(R0_rect)

    if transform_coord:
    ######### Cambiar las coordenadas de camara a LiDAR
        pointcloud = np.transpose(Tr_cam_to_velo @ R0_rect_inv @ np.transpose(pointcloud))

    print(pointcloud)


    # Plotting

    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    fig = mayavi.mlab.figure(bgcolor=(0,0,0),size=(640,500))
    mayavi.mlab.points3d(x, y, z,
                     d,          # Values used for Color
                     mode="point",
                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
                     # color=(0.5, 0.5, 0.5),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )
    mayavi.mlab.points3d(0, 0, 0,
                    mode="sphere",
                    scale_factor = 0.5,
                    colormap='spectral', # 'bone', 'copper', 'gnuplot'
                    # color=(0.5, 0.5, 0.5),   # Used a fixed (r,g,b) instead
                    figure=fig,
                    )

    #velo =  velo @ R0_rect
    x = np.asarray(velo[:, 0]).reshape(-1)  # x position of point
    y = np.asarray(velo[:, 1]).reshape(-1)  # y position of point
    z = np.asarray(velo[:, 2]).reshape(-1)  # z position of point
    r = np.asarray(velo[:, 3] ).reshape(-1) # reflectance value of point

    mayavi.mlab.points3d(x, y, z,

                     mode="point",
                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
                     color=(1, 1, 1),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )                

    f_kitti = pd.read_csv(gt_file,sep=" ",header=None)
    f_kitti.columns= ["Class","truncated","occluded","alpha","left","top","rigth","bottom","h","w","l","x","y","z","rot"]
    df_filter=f_kitti['Class']!='DontCare'
    f_kitti = f_kitti[df_filter]

    for i in range(f_kitti.shape[0]):
        plot_3d_box(f_kitti.iloc[i],fig,Tr_cam_to_velo,R0_rect_inv)


    x=np.linspace(5,5,50)
    y=np.linspace(0,0,50)
    z=np.linspace(0,5,50)
    mayavi.mlab.plot3d(x,y,z)
    mayavi.mlab.show()
    

def plot_3d_box(row,fig,Tr_cam_to_velo,R0_rect_inv):

    pos = np.array(([row['x']],[row['y']],[row['z']],[1]))
    pos = np.reshape(Tr_cam_to_velo @ R0_rect_inv @ pos, (1,4)).tolist()[0]
    # print(pos[0])
    # print(np.reshape(pos,(1,4)))
    # print(R0_rect_inv @ Tr_cam_to_velo @ pos)

    x = row['x']
    y = row['y']
    z = row['z']

    print('------------------------------')
    bbox_3d = create_3d_bbox(row['rot'],(row['h'],row['w'],row['l']),(x,y,z))
    bbox_3d = Tr_cam_to_velo @ R0_rect_inv @ bbox_3d
    bbox_3d = np.hstack((bbox_3d[:,0:4],bbox_3d[:,0],bbox_3d[:,4:],bbox_3d[:,4:6],bbox_3d[:,1:3],bbox_3d[:,6:8],bbox_3d[:,3]))

    mayavi.mlab.plot3d(bbox_3d[0,:].tolist()[0], bbox_3d[1,:].tolist()[0], bbox_3d[2,:].tolist()[0],
                     color=(1, 0, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )
    mayavi.mlab.points3d(pos[0], pos[1], pos[2],
                    mode="sphere",
                    scale_factor = 0.2,
                    colormap='spectral', # 'bone', 'copper', 'gnuplot'
                    # color=(0.5, 0.5, 0.5),   # Used a fixed (r,g,b) instead
                    figure=fig,
                    )



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
    main()