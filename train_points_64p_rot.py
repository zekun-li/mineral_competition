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
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torchvision import transforms 
import torch.optim as optim

from models import Net, Net64
from datasets import *

import argparse


np.random.seed(2333)



def validate(net, val_loader, criterion):
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        return val_loss



def train(args):
    img_json_dir = args.img_json_dir
    gt_dir = args.gt_dir
    label_map_json = args.label_map_json
    label_key_name = args.label_key_name
    checkpoint_dir = args.checkpoint_dir

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    with open(label_map_json, 'r') as f:
        label_map_dict = json.load(f)
    
    cur_map_list, cur_label_name_list = label_map_dict[label_key_name]
    # cur_map_list = cur_map_list[:2]
    # cur_label_name_list = cur_label_name_list[:2]


    transform = transforms.Compose([transforms.RandomAffine(degrees = 30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                                    transforms.Resize(64), 
                                    # transforms.RandomCrop(32),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    transform_val = transforms.Compose([transforms.Resize(64), 
                                    # transforms.RandomAffine(degrees = 0, translate=(0.3, 0.3), scale=(0.7, 1)),
                                    # transforms.RandomCrop(32),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    dataset = RawImageDataset(img_json_dir, gt_dir, map_list = cur_map_list, label_name_list = cur_label_name_list, transform=transform)
    # pdb.set_trace()

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8) ])
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers = 5, shuffle=True) # 284
    val_loader = DataLoader(val_dataset, batch_size = 16, num_workers = 5) # 72

    net = Net64()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    best_val_loss = 1000
    for epoch in range(2000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(loss.item())
            
        val_loss = validate(net, val_loader, criterion)
        print('train_loss', running_loss/len(train_dataset) ,'val_loss', val_loss/len(val_dataset))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'model_'+ label_key_name +'_best.pth'))

    torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'model_'+ label_key_name +'_last.pth'))

    print('Finished Training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_json_dir', type=str, default='/data2/mineral_competition/data/train_input')
    parser.add_argument('--gt_dir', type=str, default='/data2/mineral_competition/data/train_output')
    parser.add_argument('--label_map_json', type=str, default='../data/pointsymbols.json')
    parser.add_argument('--checkpoint_dir', type=str, default = '/data2/mineral_competition/zekun_models/checkpoints_64p_rot/')
    parser.add_argument('--label_key_name', type=str, default=None) # button, plus

    
    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(args)

if __name__ == '__main__':
    main()