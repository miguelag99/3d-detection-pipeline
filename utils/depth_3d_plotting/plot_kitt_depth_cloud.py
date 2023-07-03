import numpy as np
import os
import mayavi.mlab
import math
import pandas as pd

from generate_lidar_from_depth import depth_2_pointcloud

ROOT_PATH = '/home/robesafe/3d-detection-pipeline'

DEPTH_PATH = os.path.join(ROOT_PATH,'results/SDN_kitti/depth_maps/')

DATASET_PATH = "/media/robesafe/SSD_SATA/KITTI_DATASET/training/"
# VELO_PATH = os.path.join(ROOT_PATH,'results/SDN_kitti/lidar_from_depth/')
VELO_PATH = os.path.join(DATASET_PATH,"velodyne/")
# VELO_PATH = '/home/robesafe/3d-detection-pipeline/results/SDN_shift/lidar_from_depth/'

CALIB_PATH = os.path.join(DATASET_PATH,"calib/")
GT_PATH = os.path.join(DATASET_PATH,'label_2/')



def main(invert_y_3d_axis = False):

    # Load and transform data

    # 5(pedestrian 25m wrong),8(cars with noise),28(close pedestrian),47(multiple cars)

    id = 2
    assert os.path.isfile(DEPTH_PATH + '{:06d}.npy'.format(id)), 'Depth file not found'

    # depth_files = [DEPTH_PATH + name for name in sorted(os.listdir(DEPTH_PATH)) if name.endswith('.npy')]
    # im_name = depth_files[id].split('/')[-1].split('.')[0]

    im_name = '{:06d}'.format(id)

    gt_file = os.path.join(GT_PATH, im_name + '.txt')
    calib_file = os.path.join(CALIB_PATH, im_name + '.txt')
    velo_file = os.path.join(VELO_PATH, im_name + '.bin')
    depth_file = os.path.join(DEPTH_PATH, im_name + '.npy')
    res_file = '/home/robesafe/perception-fusion/results/sdn_kitti_50_ep/objects/000067.txt'
    
    pointcloud = depth_2_pointcloud(calib_file,depth_dir=depth_file,max_high=4).reshape([-1,4])
    velo = np.fromfile(velo_file,dtype=np.float32, count=-1).reshape([-1,4])

    print(np.max(pointcloud[:,0]))
    print(np.max(velo[:,0]))
    
    print(f'Shape velo:{velo.shape} and shape depth:{pointcloud.shape} of image {im_name}')
    print(pointcloud)

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

    print(f'P2:\n{p2}\nR0_rect:\n{R0_rect}\nTr_velo_to_cam:\n{Tr_velo_to_cam}\nTr_cam_to_velo:\n{Tr_cam_to_velo}\nR0_rect_inv:\n{R0_rect_inv}')

    # Plotting

    plot_filter = (pointcloud[:,0] > 0) & (pointcloud[:,1] > -10) & (pointcloud[:,1] < 10) & (pointcloud[:,2] < 1) 

    x = pointcloud[:, 0]  # x position of point
    if invert_y_3d_axis:
        y = -pointcloud[:, 1]
    else:
        y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    fig = mayavi.mlab.figure(bgcolor=(0,0,0),size=(1920,1080))
    mayavi.mlab.points3d(x[plot_filter], y[plot_filter], z[plot_filter],
                     d[plot_filter],          # Values used for Color
                     mode="point",
                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
                     color=(0.5, 0.5, 0.5),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )
    mayavi.mlab.points3d(0, 0, 0,
                    mode="sphere",
                    scale_factor = 0.5,
                    colormap='spectral', # 'bone', 'copper', 'gnuplot'
                    # color=(0.5, 0.5, 0.5),   # Used a fixed (r,g,b) instead
                    figure=fig,
                    )

    # velo =  velo @ R0_rect


    x = np.asarray(velo[:, 0]).reshape(-1)  # x position of point
    if invert_y_3d_axis:
        y = -np.asarray(velo[:, 1]).reshape(-1)
    else:
        y = np.asarray(velo[:, 1]).reshape(-1)  # y position of point
    z = np.asarray(velo[:, 2]).reshape(-1)  # z position of point
    r = np.asarray(velo[:, 3] ).reshape(-1) # reflectance value of point


    # mayavi.mlab.points3d(x[(x>0)&(y>-8)&(y<10)], y[(x>0)&(y>-8)&(y<10)], z[(x>0)&(y>-8)&(y<10)],
    mayavi.mlab.points3d(x, y, z,
                     mode="point",
                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
                     color=(1, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )                

    f_kitti = pd.read_csv(gt_file,sep=" ",header=None)
    f_kitti.columns= ["Class","truncated","occluded","alpha","left","top","rigth","bottom","h","w","l","x","y","z","rot"]
    df_filter=f_kitti['Class']!='DontCare'
    f_kitti = f_kitti[df_filter]

    # for i in range(f_kitti.shape[0]):
    #     plot_3d_box(f_kitti.iloc[i],fig,Tr_cam_to_velo,R0_rect_inv,invert_y_3d_axis)


    # f_kitti = pd.read_csv(res_file,sep=" ",header=None)
    # f_kitti.columns= ["Class","truncated","occluded","alpha","left","top","rigth","bottom","h","w","l","x","y","z","rot"]
    # df_filter=f_kitti['Class']!='DontCare'
    # f_kitti = f_kitti[df_filter]

    # for i in range(f_kitti.shape[0]):
    #     plot_3d_box(f_kitti.iloc[i],fig,Tr_cam_to_velo,R0_rect_inv,invert_y_3d_axis,color=(0,1,0))

    # x=np.linspace(0,50,6)
    # y=np.linspace(0,0,6)
    # z=np.linspace(0,0,6)
    # print(x)
    # mayavi.mlab.points3d(x,y,z,
    #                      mode="sphere",
    #                     scale_factor = 0.5,
    #                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
    #                     figure=fig)
    mayavi.mlab.show()
    

def plot_3d_box(row,fig,Tr_cam_to_velo,R0_rect_inv,invert_y_3d_axis, color = (1,0,0)):

    pos = np.array(([row['x']],[row['y']],[row['z']],[1]))
    pos = np.reshape(Tr_cam_to_velo @ R0_rect_inv @ pos, (1,4)).tolist()[0]

    if invert_y_3d_axis:
        pos[1] = -pos[1]

    # print(pos[0])
    # print(np.reshape(pos,(1,4)))
    # print(R0_rect_inv @ Tr_cam_to_velo @ pos)

    x = row['x']
    y = row['y']
    z = row['z']

    bbox_3d = create_3d_bbox(row['rot'],(row['h'],row['w'],row['l']),(x,y,z))
    bbox_3d = Tr_cam_to_velo @ R0_rect_inv @ bbox_3d
    bbox_3d = np.hstack((bbox_3d[:,0:4],bbox_3d[:,0],bbox_3d[:,4:],bbox_3d[:,4:6],bbox_3d[:,1:3],bbox_3d[:,6:8],bbox_3d[:,3]))

    if invert_y_3d_axis:
        bbox_3d[1,:] = -bbox_3d[1,:]

    mayavi.mlab.plot3d(bbox_3d[0,:].tolist()[0], bbox_3d[1,:].tolist()[0], bbox_3d[2,:].tolist()[0],
                     color=color,   # Used a fixed (r,g,b) instead
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