import numpy as np
import os 
from PIL import Image
import cv2
import json 
import pdb

from torch.utils.data import Dataset, DataLoader


def crop_resize(img, min_x, max_x, min_y, max_y, standard_w, standard_h):
    cropped_img = img[min_x:max_x, min_y:max_y, :]
    cropped_img = cv2.resize(cropped_img, (standard_w, standard_h))
    return cropped_img

def cv_ops(img1):
    img1 = cv2.medianBlur(img1,5)
    
    
    ch1 = cv2.adaptiveThreshold(img1[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,21,2)
    ch2 = cv2.adaptiveThreshold(img1[:,:,1],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,21,2)
    ch3 = cv2.adaptiveThreshold(img1[:,:,2],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,21,2)
    th1 = cv2.bitwise_not(ch1)
    th2 = cv2.bitwise_not(ch2)
    th3 = cv2.bitwise_not(ch3)
    
    # out_img = img1.copy()
    img1[:,:,0] = ch1
    img1[:,:,1] = ch2
    img1[:,:,2] = ch3

    return img1

class RawImageDataset(Dataset):
    def __init__(self, img_json_dir, gt_dir, map_list, label_name_list = None, transform = None):
        # if label_name_list is None:
        #     label_name_list = ['horizontal_pt', 'horizbed_pt','horiz_bedding_pt','bedding_horizontal_pt']
        
        self.standard_w = 100
        self.standard_h = 100
        self.label_name_list = label_name_list
        self.transform = transform

        dataset_dict = dict()
        for file_name, label_name in zip(map_list, label_name_list):
            filename=file_name.replace('.tif', '')
            file_path=os.path.join(img_json_dir, file_name)
            json_path=file_path.replace('.tif', '.json')

            map_dict = self.read_and_process_file(filename, file_path, json_path,label_name, gt_dir)
             
            if map_dict is not None:
                dataset_dict[filename] = map_dict

        print(len(map_list), 'maps available, load %d maps' % len(dataset_dict))
        self.dataset_dict = dataset_dict


    def read_and_process_file(self,filename, img_path, json_path, label_name, gt_dir):
        # load image into an array
        im=cv2.imread(img_path)
        im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # processed_img = cv_ops(im)
        processed_img = im 

        foreground_mask = np.bitwise_and(np.bitwise_and(im[:,:,0] == 255, im[:,:,1] == 255), im[:,:,2] == 255) 

        # read the legend annotation file
        with open(json_path,'r') as f:
            data = json.load(f)
        
        # print(json_path)
        for shape in data['shapes']:# assume only one such symbol in this map, so return after finding it
            # -------------- get template ---------------------
            label = shape['label']
            typ = label.split('_')[-1]

            if typ != 'pt':
                continue 

            if label != label_name: # 
                continue 
            else: # gather gt x,y coordinates

                points = shape['points']
                if len(points) != 2:
                    print('abnormal point format', img_path, label, points)
                    continue 

            
                xy_min, xy_max = points
                
                x_min, y_min = xy_min
                x_max, y_max = xy_max
                
                template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
                h, w = template.shape[0], template.shape[1]

                if h == 0 or w == 0:
                    print('invalid template symbol size', img_path, label, points)
                    continue 

                # template = cv2.resize(template, (self.standard_w, self.standard_h))

                # ------------ get gt_tif ----------------
                gt_tif = os.path.join(gt_dir, filename + '_' + label + '.tif')
                # print(gt_tif)

                gt_im=cv2.imread(gt_tif)[:,:,0]
                gt_x_list, gt_y_list = np.where(gt_im == 1)

                # # generate mask image where GT region and surrounding eight tiles are masked out # 0: maskout 1: keep
                mask_img = np.ones((im.shape[0], im.shape[1]))
                for gt_x, gt_y in zip(gt_x_list, gt_y_list):
                    mask_img[gt_x - int(1.5*h) : gt_x + int(1.5*h), gt_y - int(1.5*w): gt_y + int(1.5*w)] = 0 

                mask_img = np.bitwise_and(foreground_mask, mask_img) # select patches from foregrond
                # random_x_list, random_y_list = np.where(mask_img == 1)

                del im 
                # del mask_img 
                del gt_im 
            
        return {'gt_x_list':gt_x_list, 'gt_y_list':gt_y_list, # 'gt_tif':gt_tif,
                'template':template, 'label_name':label_name, 'img_path': img_path,  'processed_img':processed_img, 'mask_img':mask_img,
                # 'random_x_list':random_x_list, 'random_y_list':random_y_list}
                }

            
    def __getitem__(self, index):

        dataset_dict = self.dataset_dict 
        select_map_name = np.random.choice(list(dataset_dict))
        map_dict = dataset_dict[select_map_name]
        standard_h = self.standard_h
        standard_w = self.standard_w

        gt_x_list = map_dict['gt_x_list']
        gt_y_list = map_dict['gt_y_list']
        template = map_dict['template']
        processed_img = map_dict['processed_img']
        # img_path = map_dict['img_path']

        # im=cv2.imread(img_path)
        # im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # processed_img = cv_ops(im)
        # del im 
        
        
        # processed_img = map_dict['processed_img']
        


        th, tw = template.shape[0], template.shape[1]

        
        select_index = np.random.choice(range(0, len(gt_x_list)))
        cx = gt_x_list[select_index]
        cy = gt_y_list[select_index]
        
        up = int(th/2)
        down = th - up
        
        left = int(tw/2)
        right = tw - left
        
        if np.random.uniform() < 1/3:
        
            positive_patch = crop_resize(processed_img, cx - up ,cx + down, cy - left, cy + right, self.standard_w, self.standard_h)

            if self.transform:
                positive_patch = Image.fromarray(positive_patch)
                positive_patch = self.transform(positive_patch)

            del processed_img
            return positive_patch, 1
        else:
            
            if np.random.uniform() < 1/10:
                temp = np.random.uniform()
                if temp < 0.25:
                    negative_patch = crop_resize(processed_img, cx - up -th ,cx + down -th, cy - left, cy + right, standard_w, standard_h)
                elif temp < 0.5:
                    negative_patch = crop_resize(processed_img, cx - up +th ,cx + down +th, cy - left, cy + right, standard_w, standard_h)
                elif temp < 0.75:
                    negative_patch = crop_resize(processed_img, cx - up  ,cx + down, cy - left -tw, cy + right -tw, standard_w, standard_h)
                else:
                    negative_patch = crop_resize(processed_img, cx - up  ,cx + down, cy - left + tw, cy + right + tw, standard_w, standard_h)

            # negative_patch = np.random.choice([cropped_neg1, cropped_neg2, cropped_neg3, cropped_neg4])

            else:
                random_x_list, random_y_list = np.where(map_dict['mask_img'] == 1) 
                # random_x_list = map_dict['random_x_list']
                # random_y_list = map_dict['random_y_list']
                # random_index = np.random.choice(range(0, len(random_x_list)))
                random_index = np.random.randint(0, len(random_x_list))
                random_x = random_x_list[random_index]
                random_y = random_y_list[random_index]
                # random_x = np.random.choice(range(0, int(2./3 * processed_img.shape[0]) - th))
                # random_y = np.random.choice(range(0, int(2./3 * processed_img.shape[1]) - tw))

                negative_patch = crop_resize(processed_img, random_x, random_x + th, random_y, random_y + tw, self.standard_w, self.standard_h)

                
            if self.transform:
                negative_patch = Image.fromarray(negative_patch)
                negative_patch = self.transform(negative_patch)

            del processed_img
            return negative_patch, 0
        

    def __len__(self):
        return 500 # dummy value


class MySVCDataset(Dataset):
    def __init__(self, feat_array, labels, transform=None):
        
        self.feat_array = feat_array
        self.labels = labels
        
        self.transform = transform
        
    def __getitem__(self, index):
        feat = self.feat_array[index]
        label = self.labels[index]
    
        if self.transform:
            feat = Image.fromarray(feat)
            feat = self.transform(feat)
            
            
        return feat, label
            
    
    def __len__(self):
        return len(self.feat_array)

