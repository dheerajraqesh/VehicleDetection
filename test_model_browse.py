import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import numpy as np

def browse_and_test_image():
    output_dir = "tested_results"
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ],
        initialdir="E:/Vehicle Occlusion/bdd_images/test"
    )
    
    if not image_path:
        print("No image selected. Exiting...")
        return
    
    image_name = os.path.basename(image_path)
    
    results = model(image_path)
    
    result = results[0]
    
    filtered_boxes = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id != 7:
            filtered_boxes.append(box)
    
    result.boxes = filtered_boxes
    
    img_with_boxes = result.plot()
    
    output_path = os.path.join(output_dir, f"test_result_{image_name}")
    cv2.imwrite(output_path, img_with_boxes)
    
    print(f"\nTesting image: {image_name}")
    print("\nDetections (traffic signs ignored):")
    for box in filtered_boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"- {class_name}: {confidence:.2f}")
    
    print(f"\nResult saved as: {output_path}")

if __name__ == "__main__":
    browse_and_test_image() 