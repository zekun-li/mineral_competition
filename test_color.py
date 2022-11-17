
import sys
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from statistics import mode
from test_dnn import write_output, write_to_tif
from const_test import *

IF_DEBUG = False

point_validation_folder = '/data2/mineral_competition/data_test/TestLabels'
input_img_dir = '/data2/mineral_competition/data_test/eval_data_perfomer'
if IF_DEBUG:
    out_dir = '/data2/mineral_competition/zekun_test/color_visualize'
else:
    out_dir = '/data2/mineral_competition/zekun_test/color'

standard_h = standard_w = 100



def get_files_in_folder(point_folder):
    img_path_list = glob.glob(point_folder + '/*_pt.jpeg')
    img_path_list = sorted(img_path_list)
    return img_path_list

def read_images(img_path_list):
    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (standard_w, standard_h))
        img_list.append(img)
        
    return img_list   

def find_foreground1(img):
    th = cv2.adaptiveThreshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,101,2)
    foreground_mask = np.bitwise_not(th)
    
    kernel = np.ones((5,5),np.uint8)
    foreground_mask = cv2.erode(foreground_mask,kernel,iterations = 1)
    
    return foreground_mask


def highlight_block(start_h, end_h, start_w, end_w, img, fill_value = 255):
    # ret_img = img.copy()
    img[start_h:end_h, start_w:end_w] = fill_value
    return img

def morph_ops(mask, dilate_kernel = 30):
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    kernel = np.ones((dilate_kernel,dilate_kernel),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_contours(mask):
    ret, thresh = cv2.threshold(mask, 200, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours




val_img_path_list = get_files_in_folder(point_validation_folder)
val_img_list = read_images(val_img_path_list)

i = 0
for color_idx in color_indices:
    template_path = val_img_path_list[color_idx]
    print(template_path)
    label_name = os.path.basename(template_path).split('_label_')[1].split('.')[0] 
    img_name = os.path.basename(template_path).split('_label_')[0]
    out_file_path = os.path.join(out_dir, img_name + '_' + label_name + '.tif')
    
    

    img_path = os.path.join(input_img_dir, img_name + '.tif')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pred_img = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
        
    temp_img = val_img_list[color_idx]
    
    
    fore_mask = find_foreground1(temp_img)
    
    fore_img = temp_img[np.where(fore_mask == 255)]


    r = int(mode(fore_img[:,0]))
    g = int(mode(fore_img[:,1]))
    b = int(mode(fore_img[:,2]))
            
    # print(r,g,b)
    plt.imshow(temp_img)
    plt.show()
    sought=[r, g, b]
    
    
    color_range=20
    lower = np.array([max(0,r - color_range),max(0,g - color_range),max(0,b - color_range)])
    upper = np.array([min(255, r+color_range), min(255, g+color_range), min(255, b+color_range)])
            
    print(lower, upper)
    mask = cv2.inRange(img, lower, upper)
    
    mask = morph_ops(mask)
    
    contours = find_contours(mask)
    # print(contours)
    
    num_detects = len(contours)
    for contour in contours:
        c_w = int(np.mean(contour[:,:,0]))
        c_h = int(np.mean(contour[:,:,1]))
    
        # pred_img = highlight_block(c_h-25, c_h+25, c_w-25, c_w+25, pred_img, fill_value = 255) 
        if IF_DEBUG:
            pred_img = highlight_block(c_h-25, c_h+25, c_w-25, c_w+25, pred_img, fill_value = 255)
        else:
            pred_img = write_output(c_h-25, c_h+25, c_w-25, c_w+25, pred_img, fill_value = 1)
    
    # print('\n\n\n')
    # cv2.imwrite(out_file_path, pred_img)
    # print(out_file_path)
    write_to_tif(out_file_path, pred_img)




