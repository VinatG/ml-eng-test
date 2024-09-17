'''
Python script to define the data-loader to load the CubiCasa5k dataset for training Mask-RCNN model.
'''
import os
import sys
sys.path.append(os.path.abspath('CubiCasa5k'))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes, box_area
from floortrans.loaders.house import House
from torchvision import transforms

class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file):
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]) # Image normalization
        self.apply_transform = True
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        self.data_folder = data_folder
        # Loading txt file
        text_file_path = os.path.join(data_folder, data_file)
        self.folders = np.genfromtxt(text_file_path, dtype = 'str')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        image, target = self.get_data(index)

        if self.apply_transform:
            image = self.transform(image)
        return image, target

    def get_data(self, index):
        fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, _ = fplan.shape
        
        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)

        # Accessing only the wall labels that contains annotation for walls as well as the rooms
        wall_labels = torch.tensor(house.walls)
        wall_instance_ids = torch.tensor(house.wall_ids)
        distinct_wall_instance_ids = torch.unique(wall_instance_ids)
        
        # Ignoring the 0th indez since it corresponds to the background
        distinct_wall_instance_ids = distinct_wall_instance_ids[1:]

	# Creating binary mask for each instance of each class, then creating the corresponding the box labels
        masks = (wall_instance_ids == distinct_wall_instance_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks) # Converting the mask to box coordinates
        
        non_empty_indices = torch.where(box_area(boxes) > 0)
        final_masks = masks[non_empty_indices]
        final_boxes = boxes[non_empty_indices]

        labels = torch.ones((len(final_boxes),), dtype=torch.int64)
        for i in range(len(final_masks)):
            rows, cols = np.where(final_masks[i])
            labels[i] = wall_labels[rows[0], cols[0]]

        target = {}
        target["masks"] = final_masks 
        target["boxes"] = final_boxes  
        target["labels"] = labels  

        return fplan, target

