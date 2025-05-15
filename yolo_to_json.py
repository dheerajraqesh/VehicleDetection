import os
import json
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import numpy as np

# Default attributes for each detection
DEFAULT_ATTRIBUTES = {
    "occluded": False,
    "truncated": False,
    "trafficLightColor": "none"
}

def draw_minimal_bbox(img, x1, y1, x2, y2, class_name, color=(0, 255, 0)):
    """Draw a minimal bounding box with class name"""
    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # Draw thin rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Add class name with small font and background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    
    # Draw text
    cv2.putText(img, class_name, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return img

def generate_json_for_image(image_path):
    # Create output directories
    output_dir = "yolo_json_output"
    img_input_dir = "yolo_img_input"
    img_output_dir = "yolo_img_output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_input_dir, exist_ok=True)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Load YOLO model
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    # Copy image to yolo_img_input
    import shutil
    try:
        shutil.copy2(image_path, img_input_dir)
    except Exception as e:
        print(f"Failed to copy {image_path} to {img_input_dir}: {e}")
    
    # Get image name and load image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Run YOLO detection
    results = model(image_path)
    result = results[0]
    
    objects = []
    # Process each detection
    for idx, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        # Draw bbox on image
        img = draw_minimal_bbox(img, x1, y1, x2, y2, class_name)
        
        # Create JSON object
        obj = {
            "category": class_name,
            "id": idx,
            "attributes": DEFAULT_ATTRIBUTES.copy(),
            "box2d": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        objects.append(obj)
    
    # Save annotated image
    img_output_path = os.path.join(img_output_dir, f"{image_name}.jpg")
    cv2.imwrite(img_output_path, img)
    print(f"Saved annotated image: {img_output_path}")
    
    # Create and save JSON
    json_data = {
        "name": image_name,
        "frames": [
            {
                "timestamp": 10000,
                "objects": objects
            }
        ],
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "dawn/dusk"
        }
    }
    
    json_output_path = os.path.join(output_dir, f"{image_name}.json")
    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved JSON: {json_output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run for a single image passed as argument (pipeline mode)
        generate_json_for_image(sys.argv[1])
    else:
        # Fallback to browse and select images via dialog (manual mode)
        import tkinter as tk
        from tkinter import filedialog
        def browse_and_generate_json():
            root = tk.Tk()
            root.withdraw()
            image_paths = filedialog.askopenfilenames(
                title="Select image(s) to run YOLO and generate JSON",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png"),
                    ("All files", "*.*")
                ],
                initialdir="E:/Vehicle Occlusion/bdd_images/test"
            )
            if not image_paths:
                print("No images selected. Exiting...")
                return
            for image_path in image_paths:
                generate_json_for_image(image_path)
        browse_and_generate_json()
