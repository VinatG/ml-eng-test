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

CLASSES = ["Background", "Wall" ,"Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]

# Defining the class names and colors for the sub_classes of the rooms class
ROOM_COLORS = {
    #1: (255, 0, 0),      # Bright Red
    2: (255, 165, 0),      # Orange
    3: (0, 0, 255),      # Bright Blue
    4: (255, 255, 0),    # Yellow (Flashy but distinct)
    5: (255, 0, 255),     # Magenta
    6: (255, 105, 180),  # Hot Pink
    7: (0, 255, 255),    # Aqua
    8: (128, 0, 128),    # Purple
    9: (240, 230, 140),  # Khaki (Light and flashy)
    10: (0, 128, 128)    # Teal
}
WALL_COLOR = (255, 0, 0)  # Red color for walls

# Function to apply non-max suppression
def apply_nms(boxes, scores, iou_threshold = 0.25):
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    
    keep_indices = nms(boxes, scores, iou_threshold)
    return keep_indices

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores, thickness = 5, is_rooms = False):
    image = np.array(image)
    for box, label, score in zip(boxes, labels, scores):
        if is_rooms and label in ROOM_COLORS:
            color = ROOM_COLORS[label]  # Use the room color dictionary to fetch color for each class type
        elif not is_rooms and label == 1:  
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
    model.roi_heads.score_thresh = 0.25
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    image_tensor = image_tensor / 255.0

    with torch.no_grad():
        result = model(image_tensor)[0]
        

    boxes = result['boxes'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    print(labels)


    # Apply NMS to the predictions
    keep_indices = apply_nms(boxes, scores)
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    if target_class == "walls":
        selected_boxes = boxes[labels == 1]
        selected_labels = labels[labels == 1]
        selected_scores = scores[labels == 1]
        visualized_image = draw_boxes(image, selected_boxes, selected_labels, selected_scores, thickness = 5, is_rooms = False)
    elif target_class == "rooms":
        room_classes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        selected_boxes = boxes[np.isin(labels, room_classes)]
        selected_labels = labels[np.isin(labels, room_classes)]
        selected_scores = scores[np.isin(labels, room_classes)]
        visualized_image = draw_boxes(image, selected_boxes, selected_labels, selected_scores, thickness = 5, is_rooms = True)

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
