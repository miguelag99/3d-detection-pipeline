import os

import torch
import torchvision.transforms as T
from typing import List

from src.stereonet.model import StereoNet
from src.stereonet import stereonet_types as st
import src.stereonet.utils as utils
from dataloader import KITTIDataset


DATAPATH = '/home/robesafe/Datasets/kitti_pseudolidar/training'
ROOT_PATH = '/home/robesafe/Miguel/3d-detection-pipeline'
SAVE_PATH = os.path.join(ROOT_PATH,'results/stereonet')
IMAGE_LIST = os.path.join(ROOT_PATH,'imagenes.txt') 
# IMAGE_LIST = os.path.join(ROOT_PATH,'ImageSets/val.txt') 
WEIGHTS = os.path.join(ROOT_PATH,'checkpoints/StereoNet/epoch=20-step=744533.ckpt') 



def main():

    device = torch.device("cuda:0" if False else "cpu")
    model = StereoNet.load_from_checkpoint(WEIGHTS)
    model.to(device)
    model.eval()

    batch_size = 1

    val_transforms: List[st.TorchTransformer] = [utils.Rescale()]
    val_dataset = KITTIDataset(DATAPATH,IMAGE_LIST)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

    max_disp = model.candidate_disparities if model.mask else float('inf')


    # LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


    for batch_idx, sample in enumerate(val_loader):

        left_im = sample[0][0]
        right_im = sample[1][0]

        batch = {'left': left_im, 'right': right_im}
        # batch = utils.ToTensor()(numpy_batch)
        tensor_transformers = [utils.Resize((640, 960)), utils.Rescale(), utils.PadSampleToBatch()]
        for transformer in tensor_transformers:
            batch = transformer(batch)

        with torch.no_grad():
            starter.record()
            prediction = model(batch)[0].cpu().numpy()
            ender.record()
            torch.cuda.synchronize()
            print("CUDA elapsed time: ",str(starter.elapsed_time(ender)))


if __name__ == '__main__':
    main()