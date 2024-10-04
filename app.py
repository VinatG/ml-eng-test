'''
Python script to define the FastAPI for the trained Mask R-CNN model.
'''
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from fastapi.responses import StreamingResponse
import cv2
from io import BytesIO
import uvicorn
from torchvision.ops import nms
from detector_model_scripts.model import create_segmentation_model

# Classes in connsideration
CLASSES = ["Background", "Wall" ,"Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]

# Defining the class names and colors for the sub_classes of the rooms class
ROOM_COLORS = {
    2: [255, 165, 0],      # Orange
    3: [0, 0, 255],      # Bright Blue
    4: [255, 255, 0],    # Yellow (Flashy but distinct)
    5: [255, 0, 255],     # Magenta
    6: [255, 105, 180],  # Hot Pink
    7: [0, 255, 255],    # Aqua
    8: [128, 0, 128],    # Purple
    9: [240, 230, 140],  # Khaki (Light and flashy)
    10: [0, 128, 128]    # Teal
}
WALL_COLOR = [255, 0, 0]  # Red color for walls

# Reading the model
model_path = "detector_model_scripts/checkpoints/best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_segmentation_model(len(CLASSES))
model.load_state_dict(torch.load(model_path, map_location = device))
model.roi_heads.score_thresh = 0.25
model.to(device)
model.eval()

app = FastAPI()

# Function to apply NMS
def apply_nms(boxes, scores, iou_threshold = 0.25):
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep_indices.numpy()

# Function to draw bounding boxes
def draw_boxes(image, boxes, labels, class_type):
    for box, label in zip(boxes, labels):
        if class_type == 'room' and label in ROOM_COLORS:
            color = ROOM_COLORS[label][::-1]
        elif class_type == 'wall' and label == 1:
            color = WALL_COLOR[::-1]
        else:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
    return image

# Function to add a horizontally stacked legend for the rooms class as done in visualize.py script for the user to understand better.
def add_legend(image): 
    h, w, _ = image.shape  
    legend_width = int(0.2 * w)  # Setting the legend width as 20% of the image width
    legend = np.ones((h, legend_width, 3), dtype = np.uint8) * 255  

    font_scale = h / 1500  
    y_offset = int(h / 25)  # Space between legend items
    box_height = int(h / 40)  # height for the color box
    box_width = int(legend_width / 5)  # width for the color box

    # Class names and colors from the ROOM_COLORS list
    for idx, (label, color) in enumerate(ROOM_COLORS.items()):  # Using the Room Colors dictionary
        class_name = CLASSES[label]
        y_pos = 40 + idx * y_offset 
        cv2.putText(legend, f"{class_name}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1) 
        cv2.rectangle(legend, (legend_width - box_width - 10, y_pos - box_height // 2), (legend_width - 10, y_pos + box_height // 2), color[::-1], -1)

    # Horizontally stack the legend next to the image
    image_with_legend = np.hstack((image, legend))
    
    return image_with_legend

# Function to process the image and run model inference
def run_inference(image: Image.Image, class_type: str):
    global device
    image_np = np.array(image)
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    image_tensor = image_tensor / 255.0
    
    with torch.no_grad():
        try:
            result = model(image_tensor)[0]
            
        except:
            device = torch.device("cpu")
            model.to(device)

    # Extract boxes, labels, and scores from the predictions
    boxes = result['boxes'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()

    # Apply NMS before plotting the boxes
    keep_indices = apply_nms(boxes, scores, iou_threshold = 0.25)
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    # Process image based on the user's query
    if class_type == 'wall':
        selected_boxes = boxes[labels == 1]  
        selected_labels = labels[labels == 1]
        image_with_boxes = draw_boxes(image_np_bgr, selected_boxes, selected_labels, class_type)
    elif class_type == 'room':
        room_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        selected_boxes = boxes[np.isin(labels, room_labels)]
        selected_labels = labels[np.isin(labels, room_labels)]
        image_with_boxes = draw_boxes(image_np_bgr, selected_boxes, selected_labels, class_type)
        
        # Adding a legend for room detection
        image_with_boxes = add_legend(image_with_boxes)
    else:
        raise ValueError("Invalid class choice")

    return image_with_boxes

# API Endpoint for inference
@app.post("/run-inference")
async def run_inference_api(image: UploadFile = File(...), type: str = "wall", output_format: str = "png"):
    if type not in ['wall', 'room']:
        return {"error": "Invalid class choice"}
    
    # Trying to deal with image formats other than .png
    valid_formats = ['png', 'jpg', 'jpeg', 'tiff']
    if output_format.lower() not in valid_formats:
        return {"error": f"Invalid output format. Supported formats are: {valid_formats}"}

    # Reading the uploaded image
    image_bytes = await image.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Run the inference
    result_image = run_inference(image, type)

    _, img_encoded = cv2.imencode(f'.{output_format}', result_image)
    img_byte_arr = BytesIO(img_encoded.tobytes())
    img_byte_arr.seek(0)

    # Return the processed image
    return StreamingResponse(img_byte_arr, media_type = f"image/{output_format}")

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3000)

