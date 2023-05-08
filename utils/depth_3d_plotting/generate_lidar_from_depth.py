import argparse
import os
import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np
from tqdm import tqdm


def project_disp_to_depth(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def depth_2_pointcloud(calib_dir, depth_dir, max_high = 1, filter_by_gt = None):


    # assert os.path.isdir(depth_dir)
    # assert os.path.isdir(calib_dir)

    # predix = depth_dir[:-4]
    # calib_file = '{}/{}.txt'.format(calib_dir, predix)
    calib = kitti_util.Calibration(calib_dir)
    depth_map = np.load(depth_dir)

    if filter_by_gt is not None:
        depth_map = depth_map * filter_by_gt

    lidar = project_disp_to_depth(calib, depth_map, max_high)
    # pad 1 in the indensity dimension
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)


    return lidar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Lidar')
    parser.add_argument('--calib_dir', type=str,
                        default='~/Kitti/object/training/calib')
    parser.add_argument('--depth_dir', type=str,
                        default='~/Kitti/object/training/predicted_disparity')
    parser.add_argument('--save_dir', type=str,
                        default='~/Kitti/object/training/predicted_velodyne')
    parser.add_argument('--max_high', type=int, default=4)
    args = parser.parse_args()

    img_list = sorted(os.listdir(args.depth_dir))
    
    for img in tqdm(img_list):
        name = img.split('/')[-1].split('.')[0]
        tqdm.write('Processing {}'.format(img))
        depth_2_pointcloud(args.calib_dir + name + '.txt', args.depth_dir + '/' + img + 'npy'
                           , args.save_dir + '/' + img + '.npy', args.max_high)
        

    depth_2_pointcloud(args.calib_dir,args.depth_dir,args.save_dir, args.max_high)
