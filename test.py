import sys
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import requests
import shutil
import os
import rasterio
from tqdm import tqdm_notebook
import colorsys
import shutil
from imutils.object_detection import non_max_suppression
from scipy import spatial
import glob

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image 

from models import Net, Net64
from const import *

import argparse

def write_output(start_h, end_h, start_w, end_w, img, fill_value = 1):
    end_h = min(end_h, img.shape[0])
    end_w = min(end_w, img.shape[1])
    
    img[int((start_h + end_h)/2), int((start_w+end_w)/2)] = fill_value
    return img

def write_to_tif(out_file_path, output_tif):

    cv2.imwrite(out_file_path, output_tif)

    # convert the image to a binary raster .tif
    raster = rasterio.open(out_file_path)
    transform = raster.transform
    array     = raster.read(1)
    crs       = raster.crs 
    width     = raster.width 
    height    = raster.height 

    raster.close()


    # if crs is None:
    #     pass 
    # else:

    #     raster = rasterio.open(out_file_path)   
    #     array  = raster.read(1)
    #     raster.close()


    with rasterio.open(out_file_path, 'w', 
                        driver    = 'GTIFF', 
                        transform = transform, 
                        dtype     = rasterio.uint8, # pred_binary_raster.dtype, # rasterio.uint8, 
                        count     = 1, 
                        compress  = 'lzw', 
                        crs       = crs, 
                        width     = width, 
                        height    = height) as dst:

        dst.write(array, indexes=1)
        dst.close()

    print('wrote output to ',out_file_path)


def highlight_block(start_h, end_h, start_w, end_w, img, fill_value = 255):
    # ret_img = img.copy()
    img[start_h:end_h, start_w:end_w] = fill_value
    return img

def get_files_in_folder(point_folder):
    img_path_list = glob.glob(point_folder + '/*.jpeg')
    img_path_list = sorted(img_path_list)
    return img_path_list

def predict(net, template, im,transform_val, standard_h, standard_w):
    # tempalte and im are both RGB images

    img1 = template      # queryImage
    
    th, tw, _ = img1.shape
    fh, hw,_ = im.shape
    
    
    pred_img = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    output_tif = np.zeros((im.shape[0],im.shape[1])).astype(np.uint8)
    cv_img = im
    plt.imshow(cv_img)
    plt.show()
    

    for idx in range(0, fh, int(th/2)):
        print(idx)
        img2_list = []
        
        jdx_list = range(0, hw, int(tw/2))
        for jdx in jdx_list:
            img2 = cv_img[idx:idx+th, jdx:jdx+tw]
            img2 = cv2.resize(img2, (standard_w, standard_h))
            img2 = transform_val(Image.fromarray(img2))
            img2_list.append(img2)
        
        with torch.no_grad():
            pred = net(torch.stack(img2_list))
            # torch.argmax(pred[0]
            # print(pred)
        
        for ji in range(0, len(jdx_list)):
            jdx = jdx_list[ji]
            if torch.argmax(pred[ji]) == 1 and pred[ji][1] > 0.8:
                start_h, end_h, start_w, end_w = idx, idx + th, jdx, jdx + tw

                # pred_img = highlight_block(start_h, end_h, start_w, end_w, pred_img, fill_value = 255) 
                output_tif = write_output(start_h, end_h, start_w, end_w, output_tif, fill_value = 1)
        
        
    # plt.imshow(pred_img)
    # plt.show()
    return cv_img, pred_img, output_tif


def main():
    # img_path = '/data2/mineral_competition/data/training/AZ_Fredonia.tif'
    # template_path = '/home/zekun/mineral_competition/data/training_point/AZ_Fredonia_label_collapse_structure_pt.jpeg'

    # img_path = '/data2/mineral_competition/data/training/AZ_Arivaca_314329_1941_62500_geo_mosaic.tif'
    # template_path = '/home/zekun/mineral_competition/data/training_point/AZ_Arivaca_314329_1941_62500_geo_mosaic_label_3_pt.jpeg'

    key = args.key # 'dot' #button, triangle, triangular_matrix, x, crossed_downward_arrow, quarry_open_pit
    checkpoint_dir = args.checkpoint_dir
    # img_id = 2

    test_symbol_list = get_files_in_folder(args.input_symbol_dir)

    indices = index_list_dict[key]
    print(f'processing %d images' %len(indices))

    for idx in indices:
        template_path = test_symbol_list[idx] # template symbol path

        label_name = os.path.basename(template_path).split('_label_')[1].split('.')[0] 
        img_name = os.path.basename(template_path).split('_label_')[0]

        img_path = os.path.join(args.input_img_dir, img_name + '.tif')

        out_file_path = os.path.join(args.output_dir, img_name + '_' + label_name + '.tif')

        
    

        # img_path = '/data2/mineral_competition/data/validation/' + legend_path_dict[key][img_id].split('_label_')[0]+'.tif'
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img_name = os.path.basename(img_path).split('.')[0]
        # label_name = os.path.basename(template_path).split('_label_')[1].split('.')[0]
        # out_file_path = os.path.join(args.output_dir, img_name + '_' + label_name + '.tif')


        # template_path = '../data/validation_point/' + legend_path_dict[key][img_id]
        cur_template = cv2.imread(template_path)
        cur_template=cv2.cvtColor(cur_template, cv2.COLOR_BGR2RGB)

        net = Net64()
        net.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model_'+key+'_best.pth')))


        transform_val = transforms.Compose([transforms.Resize(64), 
                                        # transforms.RandomAffine(degrees = 0, translate=(0.3, 0.3), scale=(0.7, 1)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])


        standard_w = standard_h = 100

        cv_img, pred_img, output_tif = predict(net, cur_template, img, transform_val, standard_h, standard_w)

        
        write_to_tif(out_file_path, output_tif)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='dot')
    parser.add_argument('--checkpoint_dir', type=str, default='/data2/mineral_competition/zekun_models/checkpoints_64p/')
    parser.add_argument('--input_symbol_dir', type=str, default = '/home/zekun/mineral_competition/data/validation_point/')
    parser.add_argument('--input_img_dir', type=str, default = '/data2/mineral_competition/data/validation/')
    parser.add_argument('--output_dir', type=str, default='/data2/mineral_competition/zekun_outputs')
    
    # parser.add_argument('--label_key_name', type=str, default=None) # button, plus

    
    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    main()