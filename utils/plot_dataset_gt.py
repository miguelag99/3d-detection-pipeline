import numpy as np
import os
import mayavi.mlab
import math
import pandas as pd
import argparse



def main(args):

    if args.dataset == 'kitti':
        DATASET_PATH = "/media/robesafe/SSD_SATA/KITTI_DATASET/training/"
        IM_SIZE = (1242,375)
    else:
        DATASET_PATH = "/home/robesafe/Datasets/shift_dataset/training"
        IM_SIZE = (1280,800)
    
    VELO_PATH = os.path.join(DATASET_PATH,"velodyne/")
    CALIB_PATH = os.path.join(DATASET_PATH,"calib/")
    GT_PATH = os.path.join(DATASET_PATH,'label_2/')

    id  = args.id
    im_name = '{:06d}'.format(id)
    gt_file = os.path.join(GT_PATH, im_name + '.txt')
    calib_file = os.path.join(CALIB_PATH, im_name + '.txt')
    velo_file = os.path.join(VELO_PATH, im_name + '.bin')

    velo = np.fromfile(velo_file,dtype=np.float32, count=-1).reshape([-1,4])
    print(velo.shape)

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

    velo[:,3] = 1
    proj_pcl = (p2 @ R0_rect @ Tr_velo_to_cam @ velo.T).T
    proj_pcl[:,0] = proj_pcl[:,0] / proj_pcl[:,2]
    proj_pcl[:,1] = proj_pcl[:,1] / proj_pcl[:,2]
    filter_pcl = np.where((proj_pcl[:,0] < 0) | (proj_pcl[:,0] > IM_SIZE[0]) |
                          (proj_pcl[:,1] < 0) | (proj_pcl[:,1] > IM_SIZE[1]))
    velo = np.delete(velo,filter_pcl,axis=0)
    velo = np.delete(velo,velo[:,0]<0,axis=0)

    print(proj_pcl.shape)
    print(velo.shape)


    # Plotting

    fig = mayavi.mlab.figure(bgcolor=(0,0,0),size=(1920,1080))

    mayavi.mlab.points3d(0, 0, 0,
                mode="sphere",
                scale_factor = 0.5,
                colormap='spectral', # 'bone', 'copper', 'gnuplot'
                # color=(0.5, 0.5, 0.5),   # Used a fixed (r,g,b) instead
                figure=fig,
                )

    x = np.asarray(velo[:, 0]).reshape(-1)  # x position of point
    y = np.asarray(velo[:, 1]).reshape(-1)  # y position of point
    z = np.asarray(velo[:, 2]).reshape(-1)  # z position of point
    
    mayavi.mlab.points3d(x, y, z,
                color = (1,1,1),
                mode="point",
                colormap='spectral',
                figure=fig
                )   


    f_kitti = pd.read_csv(gt_file,sep=" ",header=None)
    f_kitti.columns= ["Class","truncated","occluded","alpha","left","top","rigth","bottom","h","w","l","x","y","z","rot"]
    df_filter=f_kitti['Class']!='DontCare'
    f_kitti = f_kitti[df_filter]

    for i,row in f_kitti.iterrows():
        plot_3d_box(row,fig,Tr_cam_to_velo,R0_rect_inv,p2,IM_SIZE)




    mayavi.mlab.show()

def plot_3d_box(row,fig,Tr_cam_to_velo,R0_rect_inv,P2,im_size,invert_y_3d_axis=False,color = (1,0,0)):

    pos = np.array(([row['x']],[row['y']],[row['z']],[1]))

    center_proj = P2 @ pos
    center_proj = center_proj[:2] / center_proj[2]

    if (center_proj[0] > 0) and (center_proj[0] < im_size[0]) and (center_proj[1] > 0) and (center_proj[1] < im_size[1]):

        pos = np.reshape(Tr_cam_to_velo @ R0_rect_inv @ pos, (1,4)).tolist()[0]

        if invert_y_3d_axis:
            pos[1] = -pos[1]

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
    parser = argparse.ArgumentParser(description='Plot KITTI and SHIFT dataset ground truth')
    # Add argparser arguments
    parser.add_argument('-i', '--id', type=int, help='Image id', default=0)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name', default='kitti',choices=['kitti','shift'])
    args = parser.parse_args()

    main(args)






# # Crea el objeto ArgumentParser
# parser = argparse.ArgumentParser(description='Descripción del programa.')

# # Agrega los argumentos que necesitas
# parser.add_argument('-a', '--arg1', type=int, help='Descripción del primer argumento.')
# parser.add_argument('-b', '--arg2', type=str, help='Descripción del segundo argumento.')

# # Parsea los argumentos de la línea de comandos
# args = parser.parse_args()

# # Accede a los valores de los argumentos
# if args.arg1:
#     print('El valor del primer argumento es:', args.arg1)

# if args.arg2:
#     print('El valor del segundo argumento es:', args.arg2)
