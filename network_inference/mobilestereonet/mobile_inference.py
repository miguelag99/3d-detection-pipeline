import os
# import time
# import cv2
import argparse
from tracemalloc import start
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


from torch.utils.data import DataLoader

from datasets import __datasets__
from models import __models__


DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/Miguel/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/mobilestereonet')
# IMAGE_LIST = os.path.join(ROOT_PATH,'imagenes.txt') 
IMAGE_LIST = os.path.join(ROOT_PATH,'ImageSets/val.txt') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/mobilestereonet/MSNet2D_SF_DS_KITTI2015.ckpt') 
DATA_TAG = 'pruebas'
KITTI_STEREO_BASELINE = 0.54    # distance between left and right images in meters (KITTI)



def main():

    parser = argparse.ArgumentParser(description='MobileStereoNet')
    parser.add_argument('--model', default='MSNet2D', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--colored', default=1, help='save colored or save for benchmark submission')
    args = parser.parse_args()

    # Model and checkpoint
    model = __models__[args.model](args.maxdisp)
    model = nn.DataParallel(model)
    model.cuda()
    print("Loading model {}".format(WEIGHTS))
    state_dict = torch.load(WEIGHTS)
    model.load_state_dict(state_dict['model'])

    # dataset, dataloader
    StereoDataset = __datasets__['kitti']
    test_dataset = StereoDataset(DATAPATH, IMAGE_LIST, False)
    TestImgLoader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if not os.path.isdir(SAVE_PATH + '/depth_maps/'):
        os.mkdir(SAVE_PATH + '/depth_maps/')

    total_time = 0

    for batch_idx, sample in enumerate(TestImgLoader):
        disp_est_tn, inference_time = test_sample(sample,model)
        f_name = sample['left_filename'][0].split('/')[-1].split('.')[0]
        print(f_name)

        # Update times
        total_time = total_time + inference_time
        
        # Load camera calib
        calib_file = open(os.path.join(DATAPATH,'calib',f'{f_name}.txt'),"r").readlines()
        fy = float(calib_file[2].strip('\n').strip("P2: ").split(' ')[5])

        
        input_left = sample['left'][0].data.cpu().numpy()
        prediction = disp_est_tn.data.cpu().numpy()[0]

        # This will not e necessary if the size of the input is known and always the same (add and modify code)
        original_size = Image.open(sample['left_filename'][0]).size
        # print(original_size)
        if original_size[0] < 1248:
            prediction = prediction[:,:original_size[0]]
        if original_size[1] < 384:
            prediction = prediction[:original_size[1],:]

        # Z(u,v) = (fu x b)/D(u,v), siendo fu horizontal focal lenght y b el baseline entre camaras
        predicted_depth = (fy*KITTI_STEREO_BASELINE)/prediction

        np.save(SAVE_PATH + '/depth_maps/' + f_name , predicted_depth)
        # np.save(SAVE_PATH + '/depth_maps_mob3d/' + f_name , prediction)

    
    print(f'Mean inference time: {total_time/len(TestImgLoader)} ms')

def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

@make_nograd_func
def test_sample(sample,model):

    # LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    model.eval()
    starter.record()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    ender.record()
    torch.cuda.synchronize()

    # print("CUDA elapsed time: ",str(starter.elapsed_time(ender)))

    return disp_ests[-1],starter.elapsed_time(ender)


if __name__ == '__main__':
    main()