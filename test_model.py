import os
import random
from ultralytics import YOLO
import cv2
import numpy as np

def test_random_image():
    output_dir = "tested_results"
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    test_dir = "E:/Vehicle Occlusion/bdd_images/test"
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    random_image = random.choice(test_images)
    image_path = os.path.join(test_dir, random_image)
    
    results = model(image_path)
    
    result = results[0]
    
    filtered_boxes = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id != 7:
            filtered_boxes.append(box)
    
    result.boxes = filtered_boxes
    
    img_with_boxes = result.plot()
    
    output_path = os.path.join(output_dir, f"test_result_{random_image}")
    cv2.imwrite(output_path, img_with_boxes)
    
    print(f"\nTesting image: {random_image}")
    print("\nDetections (traffic signs ignored):")
    for box in filtered_boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"- {class_name}: {confidence:.2f}")
    
    print(f"\nResult saved as: {output_path}")

if __name__ == "__main__":
    test_random_image() 