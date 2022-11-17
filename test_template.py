
import sys
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from statistics import mode
from test_dnn import write_output, write_to_tif
from const_test import * #template_matching_indices
from imutils.object_detection import non_max_suppression
from utils.ModifiedTM_cc import *

IF_DEBUG = False

point_validation_folder = '/data2/mineral_competition/data_test/TestLabels'
input_img_dir = '/data2/mineral_competition/data_test/eval_data_perfomer'
if IF_DEBUG:
    out_dir = '/data2/mineral_competition/zekun_test/template_visualize'
else:
    out_dir = '/data2/mineral_competition/zekun_test/template'




def get_files_in_folder(point_folder):
    img_path_list = glob.glob(point_folder + '/*_pt.jpeg')
    img_path_list = sorted(img_path_list)
    return img_path_list

def read_images(img_path_list):
    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_list.append(img)
        
    return img_list   


def highlight_block(start_h, end_h, start_w, end_w, img, fill_value = 255):
    # ret_img = img.copy()
    img[start_h:end_h, start_w:end_w] = fill_value
    return img


print('template matching indices',template_matching_indices)

val_img_path_list = get_files_in_folder(point_validation_folder)
val_img_list = read_images(val_img_path_list)

i = 0
for idx in template_matching_indices:
    template_path = val_img_path_list[idx]
    print(template_path)
    label_name = os.path.basename(template_path).split('_label_')[1].split('.')[0] 
    img_name = os.path.basename(template_path).split('_label_')[0]
    out_file_path = os.path.join(out_dir, img_name + '_' + label_name + '.tif')
    
    

    img_path = os.path.join(input_img_dir, img_name + '.tif')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pred_img = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
        
    temp_img = val_img_list[idx]
    
    th, tw = temp_img.shape[0], temp_img.shape[1]

    points_list = modifiedMatchTemplate(img, temp_img, "TM_CCORR_NORMED", 0.8, 500, [0,360], 10, [100,110], 10, True, True) 
    print(len(points_list))

    pred_img = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    for pred in points_list:
        coord = pred[0]
        c_w, c_h = coord

        if IF_DEBUG:
            pred_img = highlight_block(c_h, c_h+th, c_w, c_w+tw, pred_img, fill_value = 255)
        else:
            pred_img = write_output(c_h, c_h+th, c_w, c_w+tw, pred_img, fill_value = 1)


    write_to_tif(out_file_path, pred_img)
    # break




