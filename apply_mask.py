import sys
import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from test_dnn import write_to_tif

input_dir = '/data2/mineral_competition/zekun_test/add'
legend_dir = '/data2/mineral_competition/data_test/TestLabels'
mask_dir = '/data2/mineral_competition/data_test/map_region_mask'

output_dir = input_dir + '_applymask'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

legend_list = sorted(glob.glob(legend_dir + '/*point.jpeg'))
for legend_path in legend_list:
    temp_legend_path = os.path.basename(legend_path).split('.')[0].split('_label_')
    map_name,label_name = temp_legend_path[0], temp_legend_path[1]
    
    pred_path = os.path.join(input_dir, map_name + '_' + label_name + '.tif')
    
    if not os.path.isfile(pred_path):
        continue 

    mask_path = os.path.join(mask_dir, map_name + '_expected_crop_region.tif')
    # print(mask_path)

    mask_img = cv2.imread(mask_path)[:,:,0]

    x_list, y_list = np.where(mask_img == 255)
    h_min, h_max = np.min(x_list), np.max(x_list)
    w_min, w_max = np.min(y_list), np.max(y_list)

    pred_img = cv2.imread(pred_path)[:,:,0]
    x_list, y_list=np.where(pred_img == 1)


    output_img =  np.zeros((pred_img.shape[0], pred_img.shape[1])).astype(np.uint8)
    
    for cx, cy in zip(x_list, y_list):
        if cx < h_min or cx > h_max or cy < w_min or cy>w_max:
            continue 
        else:
            output_img[cx, cy] = 1 

    out_file_path = os.path.join(output_dir , os.path.basename(pred_path))
    print(out_file_path)
    write_to_tif(out_file_path, output_img)

