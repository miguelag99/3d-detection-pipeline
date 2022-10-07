
import configargparse
import os
import sys

from threading import Thread
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as im
from tqdm import tqdm

def main():

    parser = configargparse.ArgParser(description='Transform depth maps to images')
    parser.add_argument('--save_path', type=str,
        default='',
        help='path to save the images')
    parser.add_argument('--depth_path', type=str,
        default='',
        help='path to read the depth files')
    
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    depth_files = sorted(os.path.join(args.depth_path,d_file) for d_file in os.listdir(args.depth_path) if d_file.endswith('.npy'))

    # process_depth_files(depth_files[0:1],args.save_path)

    x = Thread(target=process_depth_files,
        args=(depth_files[0:int(np.floor(len(depth_files)*0.25))],args.save_path,))
    y = Thread(target=process_depth_files,
        args=(depth_files[int(np.floor(len(depth_files)*0.25)):int(np.floor(len(depth_files)*0.5))],args.save_path,))
    z = Thread(target=process_depth_files,
        args=(depth_files[int(np.floor(len(depth_files)*0.5)):int(np.floor(len(depth_files)*0.75))],args.save_path,))
    w = Thread(target=process_depth_files,
        args=(depth_files[int(np.floor(len(depth_files)*0.75)):-1],args.save_path,))

    x.start()
    y.start()
    z.start()
    w.start()

    # np.set_printoptions(threshold=sys.maxsize)
    # test_f = depth_read('/home/robesafe/Datasets/kitti_depth/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png')
    



def process_depth_files(source_depth, dest_folder):

    # Transform npy depth into rgb depth image with kitti format

    tqdm_iter = tqdm(source_depth,total=len(source_depth))

    for f_path in tqdm_iter:
       
        f = np.load(f_path)
        file_name = f_path.split('/')[-1].split('.')[0]
        
        # f[f > 100] = 0
        f[f == -1] = 0
        f = f*256.

        f = Image.fromarray(f.astype('uint16'))
    
        f.save(os.path.join(dest_folder,f_path.split('/')[-1].split('.')[0]+'.png'))



# def depth_read(filename):
#     # loads depth map D from png file
#     # and returns it as a numpy array,
#     # for details see readme.txt

#     depth_png = np.array(Image.open(filename), dtype=int)
#     # make sure we have a proper 16bit depth map here.. not 8bit!
#     assert(np.max(depth_png) > 255)
#     print(np.mean(depth_png))

#     depth = depth_png.astype(np.float) / 256.
#     depth[depth_png == 0] = -1.
#     return depth

if __name__ == '__main__':
    main()

