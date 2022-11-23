import cv2
import glob 
import os 
import numpy as np 
import pyvips
import pdb 

input_dir = '/data2/mineral_competition/zekun_test/zekun_final'

predicted_raster_list = sorted(glob.glob(input_dir + '/*pt.tif'))

for predicted_raster_path in predicted_raster_list:
    predicted_raster=pyvips.Image.new_from_file(predicted_raster_path, access="sequential").numpy()
    print(predicted_raster.shape)
    if len(predicted_raster.shape)==3:
        predicted_raster=predicted_raster[:,:,0]
    elif len(predicted_raster.shape)==2:
        predicted_raster=predicted_raster
    else:
        print('predicted_raster shape is not 3 or 2!!!')
        raise ValueError

    unique_values=np.unique(predicted_raster)
    print(unique_values)
    for item in unique_values:
        if int(item) not in [0, 1, 255]:
            print('value in predicted raster:', int(item), 'not in permissible values:', [0, 1, 255])
            raise ValueError
            
    if len(unique_values)==1:
        print('no prediction')
        

    centers = np.argwhere(predicted_raster==1) 
    print('point shape', centers.shape)
    