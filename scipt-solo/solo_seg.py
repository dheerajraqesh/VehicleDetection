import os
import cv2
import torch
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import json

# Configuration
OUTPUT_MASK_DIR = "solo/seg/img"
OUTPUT_JSON_DIR = "solo/seg/json"
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates in BDD format"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the polygon
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to BDD format
    polygon = []
    for point in approx:
        x, y = point[0]
        polygon.append([float(x), float(y), "L"])
    
    # Close the polygon
    if polygon:
        polygon.append([float(polygon[0][0]), float(polygon[0][1]), "L"])
    
    return polygon

def process_image(image_path):
    """Process a single image using Mask2Former"""
    print("\nInitializing Mask2Former model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance").to(device)
    model.eval()
    
    # Load image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_height, img_width = img.shape[:2]
    
    # Initialize output data
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_data = {
        "name": image_name,
        "frames": [{
            "timestamp": 10000,
            "objects": []
        }],
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "dawn/dusk"
        }
    }
    
    # Create colored mask for visualization
    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Process the entire image
    with torch.no_grad():
        inputs = processor(images=img_pil, return_tensors="pt").to(device)
        outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, 
            target_sizes=[(img_height, img_width)]
        )[0]
        
        segmentation = results["segmentation"].cpu().numpy()
        segments_info = results["segments_info"]
        
        # Process each detected instance
        for idx, seg_info in enumerate(segments_info):
            label_id = seg_info["label_id"]
            mask = (segmentation == seg_info["id"]).astype(np.uint8)
            
            # Convert mask to polygon
            polygon = mask_to_polygon(mask)
            if not polygon:
                continue
            
            # Calculate unique RGB color based on label_id
            r = (label_id * 100) % 255
            g = (label_id * 150) % 255
            b = (label_id * 200) % 255
            color = (r, g, b)
            colored_mask[mask > 0] = color
            
            # Add to JSON output
            obj_data = {
                "category": str(label_id),
                "id": idx,
                "attributes": {
                    "occluded": False,
                    "truncated": False,
                    "trafficLightColor": "none"
                },
                "poly2d": polygon
            }
            json_data["frames"][0]["objects"].append(obj_data)
    
    # Save colored mask
    mask_path = os.path.join(OUTPUT_MASK_DIR, f"{image_name}.png")
    cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved mask to: {mask_path}")
    
    # Save JSON
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{image_name}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved JSON to: {json_path}")

def main():
    # Create file dialog
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select image to segment",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("No image selected. Exiting...")
        return
    
    # Process the image
    process_image(image_path)
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 