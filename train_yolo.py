import torch
from ultralytics import YOLO
import psutil
import os

def log_system_info():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"CPU Cores: {psutil.cpu_count(logical=True)}")

def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

if __name__ == '__main__':
    log_system_info()
    model = YOLO("yolo8x.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    print("Starting training...")
    log_gpu_memory()
    model.train(
        data="E:/Vehicle Occlusion/data_yolo.yaml",
        epochs=20,
        batch=32,
        imgsz=640,
        device=0,
        optimizer="AdamW",
        lr0=0.0001,
        weight_decay=0.001,
        amp=True,
        cache=False,
        cos_lr=True,
        warmup_epochs=2,
        multi_scale=False,
        verbose=True,
        save_period=5,
        conf=0.5,
        iou=0.6
    )
    log_gpu_memory()
    print("Training complete!")