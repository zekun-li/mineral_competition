
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
from kneed import KneeLocator
import math

IF_DEBUG = False

point_validation_folder = '/data2/mineral_competition/data_test/TestLabels'
input_img_dir = '/data2/mineral_competition/data_test/eval_data_perfomer'
if IF_DEBUG:
    out_dir = '/data2/mineral_competition/zekun_test/template_visualize'
else:
    out_dir = '/data2/mineral_competition/zekun_test/template'



def adjust_elbow(elbow):
    thresh = math.ceil(elbow*100)/100 
    if thresh < 0.95:
        thresh += 0.02
    return thresh

def template_matching(img, temp_img, method):

    # find all the template matches in the basemap
    res1 = cv2.matchTemplate(img[:,:,0], temp_img[:,:,0],method ) #  #cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img[:,:,1], temp_img[:,:,1],method ) # eval(methods[-1])) #cv2.TM_CCOEFF_NORMED)
    res3 = cv2.matchTemplate(img[:,:,2], temp_img[:,:,2],method ) # eval(methods[-1])) #cv2.TM_CCOEFF_NORMED)

    histogram, bin_edges = np.histogram(res1, bins=256, range=(0.0, 1.0))
    
    kn = KneeLocator(bin_edges[np.argmax(histogram)+1:], histogram[np.argmax(histogram):], curve='convex', direction='decreasing')
    print(kn.knee)
    elbow0 = kn.knee
    

    # fig, ax = plt.subplots()
    # plt.plot(bin_edges[0:-1], histogram)
    # plt.title("Grayscale Histogram")
    # plt.xlabel("grayscale value")
    # plt.ylabel("pixels")
    # plt.xlim(0, 1.0)
    
    histogram, bin_edges = np.histogram(res2, bins=256, range=(0.0, 1.0))
    
    kn = KneeLocator(bin_edges[np.argmax(histogram)+1:], histogram[np.argmax(histogram):], curve='convex', direction='decreasing')
    elbow1 = kn.knee


    # fig, ax = plt.subplots()
    # plt.plot(bin_edges[0:-1], histogram)
    # plt.title("Grayscale Histogram")
    # plt.xlabel("grayscale value")
    # plt.ylabel("pixels")
    # plt.xlim(0, 1.0)
    
    histogram, bin_edges = np.histogram(res3, bins=256, range=(0.0, 1.0))
    
    kn = KneeLocator(bin_edges[np.argmax(histogram)+1:], histogram[np.argmax(histogram):], curve='convex', direction='decreasing')
    elbow2 = kn.knee

    
    # fig, ax = plt.subplots()
    # plt.plot(bin_edges[0:-1], histogram)
    # plt.title("Grayscale Histogram")
    # plt.xlabel("grayscale value")
    # plt.ylabel("pixels")
    # plt.xlim(0, 1.0)
    
    # elbow = np.max([elbow1,elbow2,elbow0])
    elbow0 = adjust_elbow(elbow0)
    elbow1 = adjust_elbow(elbow1)
    elbow2 = adjust_elbow(elbow2)
    
    
    print(elbow0, elbow1,elbow2)
    

    res = np.bitwise_and(np.bitwise_and(res1>elbow0,res2>elbow1),res3>elbow2)
    print(np.min(res1),np.max(res1))
    print(np.min(res2),np.max(res2))
    print(np.min(res3),np.max(res3))
    return res

def find_contours(mask):
    ret, thresh = cv2.threshold(mask, 200, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

        
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

    res = template_matching(img, temp_img, cv2.TM_CCORR_NORMED)

    res = res.astype(np.uint8) * 255
    print(np.unique(res))
    
    
    contours = find_contours(res)
    num_detects = len(contours)
    print(num_detects)
    pred_img = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    
    for contour in contours:
        c_w = int(np.mean(contour[:,:,0]))
        c_h = int(np.mean(contour[:,:,1]))

        if IF_DEBUG:
            pred_img = highlight_block(c_h, c_h+th, c_w, c_w+tw, pred_img, fill_value = 255)
        else:
            pred_img = write_output(c_h, c_h+th, c_w, c_w+tw, pred_img, fill_value = 1)


    write_to_tif(out_file_path, pred_img)
    # break




