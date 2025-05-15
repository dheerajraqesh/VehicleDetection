import torch
from ultralytics import YOLO
import psutil
import time

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
    model = YOLO("yolo11m.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    print("Starting training...")
    log_gpu_memory()
    model.train(
        data="E:/Vehicle Occlusion/data_yolo.yaml",
        epochs=100,
        batch=32,
        imgsz=1280,
        device=0,
        optimizer="AdamW",
        lr0=0.00001,
        weight_decay=0.001,
        mosaic=0.5,
        mixup=0.0,
        auto_augment="randaugment",
        amp=True,
        cache=False,
        patience=10,
        cos_lr=True,
        warmup_epochs=10,
        overlap_mask=True,
        multi_scale=False,
        verbose=True,
        save_period=10,
        conf=0.25,
        iou=0.45,
        rect=True,
        close_mosaic=10
    )
    log_gpu_memory()
    print("Training complete!")
