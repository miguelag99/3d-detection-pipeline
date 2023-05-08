import os
import numpy as np
from PIL import Image
import pandas as pd
import configargparse
from threading import Thread
from tqdm import tqdm

parser = configargparse.ArgParser(description='Transform depth maps to images')
parser.add_argument('--save_path', type=str,
    default='',
    help='path to save the filtered images')
parser.add_argument('--depth_path', type=str,
    default='',
    help='path to read the depth files')
parser.add_argument('--label_path', type=str,
    default='',
    help='path to read the label files with detections (KITTI format)')

args = parser.parse_args()


# Load an image and create a filtered copy
def filter_depth_by_2d_box(depth_img_path_list, label_path, save_path):
    for depth_img_path in tqdm(depth_img_path_list, total=len(depth_img_path_list)):
        # Image name
        img_name = depth_img_path.split('/')[-1].split('.')[0]

        # Read txt as csv
        gt = pd.read_csv(os.path.join(label_path,img_name+'.txt'), sep=" ", header=None, 
                         names=['type','truncated','occluded','alpha',
                                'bbox_left','bbox_top','bbox_right','bbox_bottom',
                                'height','width','length','x','y','z','rotation_y'])
        gt = gt[gt['bbox_left'] != 0]



        # Read depth image
        depth_img = Image.open(depth_img_path)
        depth_img = np.array(depth_img)
        copied_depth_img = np.zeros_like(depth_img, dtype=np.uint16)

        # Filter depth image
        for i in range(len(gt)):
            bbox = gt.iloc[i]['bbox_left'], gt.iloc[i]['bbox_top'], gt.iloc[i]['bbox_right'], gt.iloc[i]['bbox_bottom']
            copied_depth_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = depth_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Add floor
        (h,w) = depth_img.shape
        copied_depth_img[int(np.ceil(0.75*h)):,:] = depth_img[int(np.ceil(0.75*h)):,:]

        # Save filtered depth image
        copied_depth_img = Image.fromarray(copied_depth_img.astype('uint16'))
        copied_depth_img.save(os.path.join(save_path,img_name+'.png'))



def main():

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    depth_files = sorted(os.path.join(args.depth_path,d_file) for d_file in os.listdir(args.depth_path) if d_file.endswith('.png'))
    label_files = sorted(os.path.join(args.label_path,l_file) for l_file in os.listdir(args.label_path) if l_file.endswith('.txt'))

    assert len(depth_files) == len(label_files), 'Number of depth files and label files must be the same'

    x = Thread(target=filter_depth_by_2d_box,
        args=(depth_files[:int(np.ceil(len(depth_files)*0.5))],
              args.label_path,
              args.save_path,))
    y = Thread(target=filter_depth_by_2d_box,
        args=(depth_files[int(np.ceil(len(depth_files)*0.5)):],
              args.label_path,
              args.save_path,))


    x.start()
    y.start()



if __name__ == '__main__':
    main()
