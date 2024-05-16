import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import multiprocessing as mp

def train():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. Ensure you have the necessary drivers and CUDA installed.")

     # Set memory allocation configuration to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Load the model
    model = YOLO('yolov9c.pt')

    # Use DataParallel to use multiple GPUs if available
    model = nn.DataParallel(model)

    # Move the model to the first GPU
    model = model.cuda()

    # Define the path to the YAML file
    yaml_path = Path(__file__).parent / "data/data.yaml"

    # Train the model
    result = model.module.train(data=yaml_path, epochs=100, imgsz=640, batch=8, amp=True)
    print("Training completed.")

if __name__ == '__main__':
    # Ensure that this is the main module
    mp.freeze_support()  # Required for Windows
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # only run on primary GPU
    train()
