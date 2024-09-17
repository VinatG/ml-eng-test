'''
Python script to evaluate the trained model on the test dataset. We perform non-max suppression at an iou_threshold of 0.2 before evaluating the model.
Metric used: TPs, FPs, FNs, Precision, and Recall.

Arguments:
    1. --model-path : Path to the trained Mask R-CNN model
    2. --data-directory : Directory containing the test dataset on which the model will be evaluated
    3. --batch-size : Batch size of the test data loader for evaluation

Sample usage:
    python detector_model_scripts/evaluate.py --data-directory CubiCasa5k/data --model-path detector_model_scripts/checkpoints/best.pt --batch-size 4
'''

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import FloorplanSVG
from model import create_segmentation_model
import torchvision
from torchvision.ops import box_iou
from torchvision.ops import nms
from collections import defaultdict
from tqdm import tqdm


CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]

# Helps in dealing with stacking images with different sizes
def collate_fn(batch): 
    return tuple(zip(*batch))

# Function to calculate the iou of box1 and box2
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0
# Function to apply non-max suppression. Intution is that the bounding boxes of walls and rooms should no interest to a large extent.
def apply_nms(predictions, iou_threshold = 0.2):
    keep = nms(predictions['boxes'], predictions['scores'], iou_threshold)
    predictions['boxes'] = predictions['boxes'][keep]
    predictions['labels'] = predictions['labels'][keep]
    predictions['scores'] = predictions['scores'][keep]
    
    return predictions

# Fuction to calculate the metrics(We are usig Precision and Recall as the metric)
def calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold = 0.5):

    tp, fp, fn = 0, 0, 0
    used_true_boxes = set()
    
    for i, pred_box in enumerate(pred_boxes):
        pred_label = pred_labels[i]
        match_found = False
        
        for j, true_box in enumerate(true_boxes):
            if j in used_true_boxes or true_labels[j] != pred_label:
                continue
            if iou(pred_box, true_box) >= iou_threshold:
                tp += 1
                used_true_boxes.add(j)
                match_found = True
                break
                
        if not match_found:
            fp += 1
    
    fn = len(true_boxes) - len(used_true_boxes)
    return tp, fp, fn

# Function where we perform the evaluation of the model on the test dataset
def evaluate_model(model, dataloader, device, iou_threshold = 0.2):
    tp_dict = defaultdict(int)
    fp_dict = defaultdict(int)
    fn_dict = defaultdict(int)

    model.to(device)
    model.eval()
    
    for images, targets in tqdm(dataloader, desc = "Evaluating", unit = "batch"):
        # Move images and targets to the specified device
        images = [image.to(device) for image in images]
        targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

        with torch.no_grad():
            predictions = model(images)

        for i, prediction in enumerate(predictions):
            prediction = apply_nms(prediction, iou_threshold)

            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_labels = prediction['labels'].cpu().numpy()
            true_boxes = targets[i]['boxes'].cpu().numpy()
            true_labels = targets[i]['labels'].cpu().numpy()

            tp, fp, fn = calculate_metrics(pred_boxes[pred_labels == 2], pred_labels[pred_labels == 2], true_boxes[true_labels == 2], true_labels[true_labels == 2])
            tp_dict['walls'] += tp
            fp_dict['walls'] += fp
            fn_dict['walls'] += fn
            # Treating index 1 and 3 to 11 as rooms
            rooms_pred_indices = np.isin(pred_labels, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            rooms_true_indices = np.isin(true_labels, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11])

            tp, fp, fn = calculate_metrics(pred_boxes[rooms_pred_indices], pred_labels[rooms_pred_indices], true_boxes[rooms_true_indices], true_labels[rooms_true_indices])
            tp_dict['rooms'] += tp
            fp_dict['rooms'] += fp
            fn_dict['rooms'] += fn

    return tp_dict, fp_dict, fn_dict

# Function to calculate the precision given tp and fp
def precision(tp, fp):
    return tp / (tp + fp) if tp + fp > 0 else 0

# Function to calculate the recall given tp and fn
def recall(tp, fn):
    return tp / (tp + fn) if tp + fn > 0 else 0

# Fuction to print the results dictioary in an elegant way
def print_results(tp_dict, fp_dict, fn_dict):
    for class_label in tp_dict.keys():
        tp, fp, fn = tp_dict[class_label], fp_dict[class_label], fn_dict[class_label]
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        print(f"Class {class_label}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn} Precision: {prec:.4f}, Recall: {rec:.4f}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = create_segmentation_model(len(CLASSES))
    model.load_state_dict(torch.load(args.model_path, map_location = device))
    model.to(device)
    
    dataset = FloorplanSVG(args.data_directory, 'test.txt')
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    
    tp_dict, fp_dict, fn_dict = evaluate_model(model, dataloader, device)
    print_results(tp_dict, fp_dict, fn_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Evaluate Mask R-CNN Model for Precision and Recall")
    parser.add_argument("--model-path", type = str, required = True, help = "Path to the trained Mask R-CNN model")
    parser.add_argument("--data-directory", type = str, required = True, help = "Path to the directory containing the dataset")
    parser.add_argument("--batch-size", type = int, default = 1, help = "Batch size for evaluation")
    args = parser.parse_args()
    
    main(args)

