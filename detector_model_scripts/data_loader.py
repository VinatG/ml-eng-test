"""
Python script to build the Data Loader to load the walls and the other rooms classes dataset
"""
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
        self.transform = transforms.Compose([transforms.ToTensor()]) # Simply dividing the image by 255 and converting it to a tensor
        self.apply_transform = True
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        self.data_folder = data_folder
        # Load txt file to list
        text_file_path = os.path.join(data_folder, data_file)
        self.folders = np.genfromtxt(text_file_path, dtype = 'str')

    def get_target_for_walls(self, house):
        target_walls = {"masks": [], "boxes": [], "labels": []}
        wall_instance_ids = torch.tensor(house.wall_ids)
        distinct_wall_instance_ids = torch.unique(wall_instance_ids)

        distinct_wall_instance_ids = distinct_wall_instance_ids[1:]

        masks = (wall_instance_ids == distinct_wall_instance_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks) # Converting the mask to box coordinates

        non_empty_indices = torch.where(box_area(boxes) > 0)

        final_masks = masks[non_empty_indices]
        final_boxes = boxes[non_empty_indices]
        labels = torch.tensor([1 for i in range(final_boxes.shape[0])])

        target_walls["masks"] = final_masks
        target_walls["boxes"] = final_boxes
        target_walls["labels"] = labels

        return target_walls

    def get_target_for_rooms(self, mask_tensor):
        target_rooms = {"masks": [], "boxes": [], "labels": []}
        # We are taking all the classes except for Background and Outdoor
        class_mapping = { 3 : 2, 4 : 3, 5 : 4, 6 : 5, 7 : 6, 8 : 7, 9 : 8, 10 : 9, 11 : 10} 

        for class_idx, mapped_class in class_mapping.items():
            mask = (mask_tensor == class_idx).float()  # Binary mask (H, W)
            mask_np = mask.cpu().numpy().astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(mask_np)

            for instance_id in range(1, num_labels):  # Skip background, so start from 1
                instance_mask = (labels_im == instance_id).astype(np.float32)

                instance_mask_tensor = torch.from_numpy(instance_mask).to(mask.device)
                # Check if the mask has any non-zero area
                if torch.sum(instance_mask_tensor) > 0:
                    box = masks_to_boxes(instance_mask_tensor.unsqueeze(0))
                    area = box_area(box)
                    if area > 0:
                        target_rooms["masks"].append(instance_mask_tensor)
                        target_rooms["boxes"].append(box[0])
                        target_rooms["labels"].append(mapped_class)

        target_rooms["masks"] = torch.stack(target_rooms["masks"], dim = 0) if target_rooms["masks"] else torch.tensor([])
        target_rooms["boxes"] = torch.stack(target_rooms["boxes"], dim = 0) if target_rooms["boxes"] else torch.tensor([])
        target_rooms["labels"] = torch.tensor(target_rooms["labels"]) if target_rooms["labels"] else torch.tensor([])

        return target_rooms

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        image, target = self.get_data(index)

        if True: 
            image = self.transform(image)
        return image, target

    def get_data(self, index):
        
        fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        self.image = fplan
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, _ = fplan.shape

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)

        target_walls = self.get_target_for_walls(house)
        masks = torch.tensor(house.get_segmentation_tensor()[0].astype(np.float32))
        target_rooms = self.get_target_for_rooms(masks)
        target = {}
        
        if target_walls['masks'].numel() > 0 and target_rooms['masks'].numel() > 0:
            target["masks"] = torch.cat((target_walls['masks'], target_rooms['masks']), dim=0)
        elif target_walls['masks'].numel() > 0:
            target["masks"] = target_walls['masks']
        elif target_rooms['masks'].numel() > 0:
            target["masks"] = target_rooms['masks']
        else:
            target["masks"] = torch.tensor([])  # Empty tensor
    
        if target_walls['boxes'].numel() > 0 and target_rooms['boxes'].numel() > 0:
            target["boxes"] = torch.cat((target_walls['boxes'], target_rooms['boxes']), dim=0)
        elif target_walls['boxes'].numel() > 0:
            target["boxes"] = target_walls['boxes']
        elif target_rooms['boxes'].numel() > 0:
            target["boxes"] = target_rooms['boxes']
        else:
            target["boxes"] = torch.tensor([])  
    
        if target_walls['labels'].numel() > 0 and target_rooms['labels'].numel() > 0:
            target["labels"] = torch.cat((target_walls['labels'], target_rooms['labels']), dim=0)
        elif target_walls['labels'].numel() > 0:
            target["labels"] = target_walls['labels']
        elif target_rooms['labels'].numel() > 0:
            target["labels"] = target_rooms['labels']
        else:
            target["labels"] = torch.tensor([])

        return fplan, target

