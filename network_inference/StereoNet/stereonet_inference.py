import os

import numpy as np
import torch
import torchvision.transforms as T
from typing import List


from dataloader import KITTIDataset

from stereonet import utils as utils
from stereonet.model import StereoNet


DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/Miguel/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/stereonet')
IMAGE_LIST = os.path.join(ROOT_PATH,'imagenes.txt') 
# IMAGE_LIST = os.path.join(ROOT_PATH,'ImageSets/val.txt') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/StereoNet/epoch=20-step=744533.ckpt') 



def main():

    # LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with open(IMAGE_LIST,'r') as f_content:
        images_path = list(map(lambda x: x.split('\n')[0], f_content.readlines()))
        
    model = StereoNet.load_from_checkpoint(WEIGHTS)

    for image_name in images_path:
        
        left_path = os.path.join(DATAPATH, 'image_2', image_name +'.png')
        right_path = os.path.join(DATAPATH, 'image_3', image_name +'.png')

        print(left_path)
    
        sample = {'left': utils.image_loader(left_path)[:,:1000],
                'right': utils.image_loader(right_path)[:,:1000]
                }

        transformers = [utils.ToTensor(), utils.PadSampleToBatch()]
        for transformer in transformers:
            sample = transformer(sample)

        model.eval()
        with torch.no_grad():
            starter.record()
            batched_prediction = model(sample)
            ender.record()
            torch.cuda.synchronize()
            print(f'Inference time: {starter.elapsed_time(ender)} ms')
        
        single_prediction = batched_prediction[0].numpy()  # [batch, ...] -> [...]
        single_prediction = np.moveaxis(single_prediction, 0, 2)  # [channel, height, width] -> [height, width, channel]

        print(single_prediction.shape)

if __name__ == '__main__':
    main()