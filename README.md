# Machine Learning Engineer Test: Computer Vision and Object Detection

## 🚨 **LATEST UPDATES** 🚨

### 🗓 **Date: 04-10-2024**  
 -🔧 **Changes Made**:  
  - Improved DataLoader indexing and focused training on the "high_quality" folder from the CubiCasa5k dataset, repeating with 10 epochs.
  - Increased training data for better model performance and added both quantitative and qualitative results.
  - Added a legend to room visualizations, using distinct colors for different room classes.

## Objective
This test aims to assess your skills in computer vision and object detection, with a specific focus on detecting room walls and identifying rooms in architectural blueprints or pre-construction plans.

This test evaluates your practical skills in applying advanced computer vision techniques to a specialized domain and your ability to integrate machine learning models into a simple API server for real-world applications.

Good luck!

## Approach
A Mask R-CNN model was trained on a partial CubiCasa5k dataset for 20 epochs. The CubiCasa5k dataset has annotations for 12 classes: "Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", and "Undefined". We have ignored the "Background" class while training.  

For evaluation and visualization, we treat index 2 ("Wall") as the "Wall" class and all other classes (excluding "Background" and "Wall") as the "Room" class.

For more details about the training process and approach, please refer to the [report.pdf](./report.pdf)  file available in this repository.

Scripts utilized for training, evaluating, and visualizing the model are present in the [detector_model_scripts](./detector_model_scripts) directory.

The API has been implemented using FastAPI for seamless inference of room and wall detection.

## Results
| Class      | TP   | FP   | FN   | Precision | Recall  |
|------------|------|------|------|-----------|---------|
| Walls      | 3057 | 1044  | 1512 | 0.7454    | 0.6691  |
| Rooms      | 1106 | 663  | 815  | 0.6252    | 0.5757  |

## Visualization

Here are sample images from the results:

- **Input Image**:
  ![Input Image](./images/input_image.png)

- **Input Image Annotated with Predicted Walls**:
  ![Walls Prediction](./images/input_walls_annotated.png)

- **Input Image Annotated with Predicted Rooms**:
  ![Rooms Prediction](./images/input_rooms_annotated.png)

## Installation

1. Clone the repository:
```
git clone https://github.com/VinatG/ml-eng-test.git
```
2. Change the directory
```
cd ml-eng-test
```
3. Download the trained model:  
Due to size constraints, the trained model file is not included in this repository. Since this repository has been forked from another repository, Git LFS (Large File Storage) cannot be used to add larger files. You need to manually download the model using the following link:
    - [Download the trained Mask R-CNN model](https://drive.google.com/file/d/1vgacDlyUYRdtKBYkQZ6jxR94Q7DZrfD8/view?usp=sharing)

Once downloaded, place the model file in the `detector_model_scripts/checkpoints` directory.

4. Build the Docker container:
```
sudo docker build -t room-wall-detection-app .
```
5. Run the Docker container:
```
sudo docker run -d -p 3000:3000 --network="host" room-wall-detection-app
```
## Usage

### Wall Detection
Run the following curl command to perform wall detection:
```
curl -X POST -F "image=@image.png" "localhost:3000/run-inference?type=wall" --output output.png
```
### Room Detection
Run the following curl command to perform room detection:
```
curl -X POST -F "image=@image.png" "localhost:3000/run-inference?type=room" --output output.png
```


