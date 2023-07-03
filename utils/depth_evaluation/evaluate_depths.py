import os
import numpy as np
import configargparse
import pandas as pd

import pdb

from PIL import Image
from tqdm import tqdm

def compute_errors(gt, pred):

    gt = np.reshape(gt,(-1))
    pred = np.reshape(pred,(-1))

    mask = (gt > 0) & (pred > 0)
    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    irmse = ((1/(gt*1e-3))-(1/(pred*1e-3))) ** 2
    irmse = np.sqrt(irmse.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt**2)

    di = np.log(pred) - np.log(gt)
    SILog = np.mean(di**2) - np.mean(di)**2


    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, irmse, SILog

def load_depth(path):

    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32) / 256.
    return depth


def main():


    parser = configargparse.ArgParser(description='Evaluate depth estimation results')
    parser.add_argument('--gt_path',required = True, type=str,
        default='',
        help='path to read the gt depth images')
    parser.add_argument('--depth_path',required = True, type=str,
        default='',
        help='path to read the estimated depth files')
    parser.add_argument('--split_file', type=str,
        default='',
        help='path to read the estimated depth files')

    args = parser.parse_args()

    if args.split_file == '':
        gt_list = sorted(os.path.join(args.gt_path,gt_file) for gt_file 
                        in os.listdir(args.gt_path) if gt_file.endswith('.png'))
        depth_list = sorted(os.path.join(args.depth_path,depth_file) for depth_file 
                            in os.listdir(args.depth_path) if depth_file.endswith('.png'))

    else:
        with open(args.split_file,'r') as f_content:
            split_list = list(map(lambda x: x.split('\n')[0], f_content.readlines()))
        gt_list = sorted(os.path.join(args.gt_path,gt_file) for gt_file 
                        in os.listdir(args.gt_path) if gt_file.endswith('.png') and gt_file.split('.')[0] in split_list)
        depth_list = sorted(os.path.join(args.depth_path,depth_file) for depth_file 
                            in os.listdir(args.depth_path) if depth_file.endswith('.png') and depth_file.split('.')[0] in split_list)

    assert len(gt_list) == len(depth_list), 'Number of gt files and depth files must be the same'


    abs_rel_total = np.array([])
    sq_rel_total = np.array([])
    rmse_total = np.array([])
    rmse_log_total = np.array([])
    a1_total = np.array([])
    a2_total = np.array([])
    a3_total = np.array([])
    irmse_total = np.array([])
    silog_total = np.array([])


    for (depth,gt) in tqdm(zip(depth_list,gt_list),total=len(depth_list)):
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, irmse, silog = compute_errors(load_depth(gt),load_depth(depth))
        # print(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
        abs_rel_total = np.hstack((abs_rel_total,abs_rel))
        sq_rel_total = np.hstack((sq_rel_total,sq_rel))
        rmse_total = np.hstack((rmse_total,rmse))
        rmse_log_total = np.hstack((rmse_log_total,rmse_log))
        a1_total = np.hstack((a1_total,a1))
        a2_total = np.hstack((a2_total,a2))
        a3_total = np.hstack((a3_total,a3))
        irmse_total = np.hstack((irmse_total,irmse))
        silog_total = np.hstack((silog_total,silog))



    print(sq_rel_total)
    print('abs_rel: ',np.mean(abs_rel_total))
    print('sq_rel: ',np.mean(sq_rel_total))
    print('rmse: ',np.mean(rmse_total))
    print('rmse_log: ',np.mean(rmse_log_total))
    print('a1: ',np.mean(a1_total))
    print('a2: ',np.mean(a2_total))
    print('a3: ',np.mean(a3_total))
    print('irmse: ',np.mean(irmse_total))
    print('silog: ',np.mean(silog_total))

if __name__ == '__main__':
    main()