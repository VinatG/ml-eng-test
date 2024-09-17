'''
Python script to inference the trained model on an image and visualize the results.

Arguments:
    1. --model-path : Path to the trained Mask R-CNN model
    2. --image-path : Path to the image on which inference and visualization needs to be performed
    3. --output-path : Path to save the processed image
    4. --class-name : The user needs to specify "walls" or "rooms"

Sample usage:
    python detector_model_scripts/visualize.py --model-path detector_model_scripts/checkpoints/best.pt --image-path sample.png
    --output-path output.png --class-name walls
'''
import torch
import numpy as np
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms
from model import create_segmentation_model

CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", 
           "Railing", "Storage", "Garage", "Undefined"]

# Defining the class names and colors for the sub_classes of the rooms class
ROOM_COLORS = {
    1: (255, 255, 153),  # Light Yellow
    3: (128, 128, 0),    # Olive Green
    4: (128, 128, 128),  # Gray
    5: (240, 128, 128),  # Light Coral
    6: (173, 216, 230),  # Light Blue
    7: (255, 182, 193),  # Light Pink
    8: (255, 165, 0),    # Light Orange
    9: (72, 209, 204),   # Light Aqua
    10: (144, 238, 144), # Light Green
    11: (255, 218, 185)  # Peach
}

WALL_COLOR = (255, 0, 0)  # Red color for walls

# Function to apply non-max suppression
def apply_nms(boxes, scores, iou_threshold = 0.1):
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    
    keep_indices = nms(boxes, scores, iou_threshold)
    return keep_indices

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores, thickness = 2, is_rooms = False):
    image = np.array(image)
    for box, label, score in zip(boxes, labels, scores):
        if is_rooms and label in ROOM_COLORS:
            color = ROOM_COLORS[label]  # Use the room color dictionary to fetch color for each class type
        elif not is_rooms and label == 2:  
            color = WALL_COLOR
        else:
            continue  
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

# Function to add a legend for rooms
def add_legend(ax):
    for idx, (label, color) in enumerate(ROOM_COLORS.items()):
        ax.text(1.05, 0.9 - (idx * 0.05), f"{CLASSES[label]}",
                fontsize = 10, color = 'black', bbox = dict(facecolor=np.array(color) / 255, edgecolor='none'), transform = ax.transAxes)

# Function to visualize the results of the model on the input image
def visualize(model_path, image_path, output_path, target_class):
    device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_segmentation_model(len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        result = model(image_tensor)[0]

    boxes = result['boxes'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()

    # Apply NMS to the predictions
    keep_indices = apply_nms(boxes, scores)
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    if target_class == "walls":
        selected_boxes = boxes[labels == 2]
        selected_labels = labels[labels == 2]
        selected_scores = scores[labels == 2]
        visualized_image = draw_boxes(image, selected_boxes, selected_labels, selected_scores, thickness = 2, is_rooms = False)
    elif target_class == "rooms":
        room_classes = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        selected_boxes = boxes[np.isin(labels, room_classes)]
        selected_labels = labels[np.isin(labels, room_classes)]
        selected_scores = scores[np.isin(labels, room_classes)]
        visualized_image = draw_boxes(image, selected_boxes, selected_labels, selected_scores, thickness = 2, is_rooms = True)

    fig, ax = plt.subplots(1, figsize = (10, 10))
    ax.imshow(visualized_image)
    ax.axis('off')

    # Add legend if user is visualizing rooms
    if target_class == "rooms":
        add_legend(ax)

    fig.savefig(output_path, bbox_inches = 'tight', pad_inches = 0.1)

def main(args):
    visualize(args.model_path, args.image_path, args.output_path, args.class_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing the Mask R-CNN predictions")
    parser.add_argument("--model-path", type = str, required = True, help = "Path to the trained Mask R-CNN model")
    parser.add_argument("--image-path", type = str, required = True, help = "Path to the input image")
    parser.add_argument("--output-path", type = str, required = True, help = "Path to save the output image")
    parser.add_argument("--class-name", type = str, default = 'walls', choices = ["walls", "rooms"], help = "Class to visualize ('walls' or 'rooms')")
    args = parser.parse_args()
    
    main(args)

