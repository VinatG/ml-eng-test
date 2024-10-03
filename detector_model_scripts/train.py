'''
Python script to train the mask R-CNN model on the dataset.

CLI Arguements:
    1. --data-directory : Path to directory containing the CubiCasa5k dataset
    2. --batch-size : Batch size to be used
    3. --epochs : Number of epochs for which the model would be trained
    4. --print-freq : Frquency of printing the training log(print-freq = 1 corresponds to printing the logs every training step)
    5. --use-subset : If set, uses a sub-set of the training data instead of all the 4,200 images.
    6. --save-dir : Directory where the model checkpoints, training curves and the logs will be saved
    7. --save-logs : If set, saves he train and val loss logs for each epoch
    8. --lr : Learning rate
    9. --momentum : momentum for the optimizer
    10. --weight_decay : Weight decay for the optimizer
    11. --step_size : Step-size for the learning rate scheduler
    12. --gamma : Gamma for the learning rate scheduler

Sample usage:

    python detector_model_scripts/train.py --data-directory CubiCasa5k/data --batch-size 8 --epochs 20 --print-freq 10 
    --use-subset --save-dir detector_model_scripts/checkpoints --save-logs
'''

CLASSES = ["Background", "Wall" ,"Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]

from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader
from model import create_segmentation_model
import torchvision.transforms as T
import argparse
import os
import matplotlib.pyplot as plt
from data_loader import FloorplanSVG
from time import time

# Custom collate function to deal with stacking of images with different sizes
def collate_fn(batch):
    return tuple(zip(*batch))

# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq = 50):
    model.train()
    
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum([loss for loss in loss_dict.values()])

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        torch.cuda.empty_cache()

        if i % print_freq == 0:
            print(f"Epoch [{epoch + 1}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

    return running_loss / len(data_loader)

# Validation function
def evaluate(model, data_loader, device):
    model.train() # In eval mode the model returns predictions along with the loss, hence keeping it train
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum([loss for loss in loss_dict.values()])
            running_val_loss += losses.item()
            torch.cuda.empty_cache()

    return running_val_loss / len(data_loader)

# Function to plot and save the train-val loss curves 
def save_loss_graph(train_losses, val_losses, save_dir):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and Validation Loss over Epochs")
    plt.savefig(os.path.join(save_dir, "train_val_loss_graph.png"))
    plt.close()

# Function to save the train-val loss logs to a text file
def save_loss_logs(train_losses, val_losses, save_dir):
    with open(os.path.join(save_dir, "loss_logs.txt"), 'w') as f:
        f.write("Epoch\tTrain Loss\tValidation Loss\n")
        for epoch in range(len(train_losses)):
            f.write(f"{epoch + 1}\t{train_losses[epoch]:.4f}\t{val_losses[epoch]:.4f}\n")

def main(args):

    train_normal_set = FloorplanSVG(args.data_directory, 'train.txt')
    if args.use_subset:
        generator1 = torch.Generator().manual_seed(42) # So that we can recreate the results
        subset_size = 400 # 400 train, 400 val, keeping a ratio of 1 : 1
        subset_dataset, _ = random_split(train_normal_set, [subset_size, len(train_normal_set) - subset_size], generator = generator1)
        train_data_loader = DataLoader(subset_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    else:
        train_data_loader = DataLoader(train_normal_set, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)

    valid_normal_set = FloorplanSVG(args.data_directory,'val.txt')
    val_data_loader = DataLoader(valid_normal_set, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    # Load the model
    model = create_segmentation_model(num_classes = len(CLASSES))

    # Set the device as CUDA if available, else use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setting up the optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)

    # Variables to track the best model and save the loss logs for each epoch
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(args.epochs):
        t1 = time()
        train_loss = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq = args.print_freq)
        t2 = time()
        train_time = t2 - t1
        t1 = time()
        val_loss = evaluate(model, val_data_loader, device)
        t2 = time()
        eval_time = t2 - t1

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check and save for the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

        # Save the model every 3rd epoch
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"maskrcnn_model_epoch_{epoch + 1}.pth"))

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time Taken Training: {train_time:.2f} seconds, Time Taken Evaluation: {eval_time:.2f} seconds")

        lr_scheduler.step()
        save_loss_graph(train_losses, val_losses, args.save_dir)
        if args.save_logs:
            save_loss_logs(train_losses, val_losses, args.save_dir)

    # Save train-val loss plots
    save_loss_graph(train_losses, val_losses, args.save_dir)

    # Save loss logs if specified by the user
    if args.save_logs:
        save_loss_logs(train_losses, val_losses, args.save_dir)

    # Epoch with the best validation loss
    print(f"Best model saved at epoch {best_epoch + 1} with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--data-directory", type = str, required = True, help = "Path to the directory containing the data")
    parser.add_argument("--batch-size", type = int, default = 2, help = "Batch size")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs")
    parser.add_argument("--lr", type = float, default = 0.005, help = "Learning rate")
    parser.add_argument("--momentum", type = float, default = 0.9, help = "Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type = float, default = 0.0005, help = "Weight decay for SGD optimizer")
    parser.add_argument("--step_size", type = int, default = 3, help = "Step size for the learning rate scheduler")
    parser.add_argument("--gamma", type = float, default = 0.1, help = "Gamma for the learning rate scheduler")
    parser.add_argument("--print-freq", type = int, default = 50, help = "Frequency of prinitng the training log")
    parser.add_argument("--use-subset", action = "store_true", help = "If set, uses a subset of the training dataset instad of all the 4200 images")
    parser.add_argument("--save-dir", type = str, default = "./checkpoints", help = "Directory to save the model checkpoints and the training graph")
    parser.add_argument("--save-logs", action = "store_true", help = "If set, saves he train and val loss logs for each epoch")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok = True)
    main(args)
