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

# Create output directories
OUTPUT_JSON_DIR = "solo_yolo_json"
OUTPUT_IMG_DIR = "solo_yolo_img"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

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

def process_image(image_path):
    """Process a single image with YOLO and generate visualization + JSON"""
    print(f"\nProcessing image: {image_path}")
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    # Load and check image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Get image name
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Run YOLO detection
    print("Running detection...")
    results = model(image_path)
    result = results[0]
    
    objects = []
    # Process each detection
    for idx, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        print(f"Detected {class_name} with confidence: {confidence:.2f}")
        
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
    img_output_path = os.path.join(OUTPUT_IMG_DIR, f"{image_name}.jpg")
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
    
    json_output_path = os.path.join(OUTPUT_JSON_DIR, f"{image_name}.json")
    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved JSON: {json_output_path}")

def main():
    # Create and hide root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    image_path = filedialog.askopenfilename(
        title="Select image to process",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ],
        initialdir="E:/Vehicle Occlusion/bdd_images/test"
    )
    
    if not image_path:
        print("No image selected. Exiting...")
        return
    
    # Process the selected image
    process_image(image_path)
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 