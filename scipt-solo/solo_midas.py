import os
import json
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt

# Configuration
OUTPUT_JSON_DIR = "solo/midas/json"
OUTPUT_IMG_DIR = "solo/midas/img"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def generate_grid_points(width, height, grid_size):
    """Generate grid points for depth sampling"""
    x_points = np.linspace(0, width-1, grid_size, dtype=int)
    y_points = np.linspace(0, height-1, grid_size, dtype=int)
    
    points = []
    for y in y_points:
        for x in x_points:
            points.append([float(x), float(y), "L"])
    
    # Close the polygon by repeating first point
    if points:
        points.append([float(points[0][0]), float(points[0][1]), "L"])
    
    return points

def process_image(image_path):
    """Process a single image with MiDaS depth estimation"""
    print("\nInitializing MiDaS model...")
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained MiDaS model
    print("Loading MiDaS model...")
    processor = AutoImageProcessor.from_pretrained("Intel/dpt-large", use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.to(device)
    model.eval()

    # Load and check image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Get image name
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Prepare image for model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    # Run depth estimation
    print("Running depth estimation...")
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(img_height, img_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = prediction.cpu().numpy()

    # Create visualization
    # Normalize depth map for visualization
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Create colored depth visualization
    colored_depth = plt.cm.viridis(depth_norm)[:, :, :3]  # Remove alpha channel
    colored_depth = (colored_depth * 255).astype(np.uint8)
    
    # Add text with depth information
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    avg_depth = depth_map.mean()
    
    # Add depth info text to the visualization
    info_img = colored_depth.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info_img, f"Min depth: {min_depth:.2f}", (20, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, f"Max depth: {max_depth:.2f}", (20, 60), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, f"Avg depth: {avg_depth:.2f}", (20, 90), font, 0.7, (255, 255, 255), 2)
    
    # Save depth visualization
    vis_path = os.path.join(OUTPUT_IMG_DIR, f"{image_name}.png")
    cv2.imwrite(vis_path, cv2.cvtColor(info_img, cv2.COLOR_RGB2BGR))
    print(f"Saved depth visualization to: {vis_path}")
    
    # Create simple blend with original image
    alpha = 0.7
    blend = cv2.addWeighted(img, 1-alpha, cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR), alpha, 0)
    blend_path = os.path.join(OUTPUT_IMG_DIR, f"{image_name}_blend.png")
    cv2.imwrite(blend_path, blend)
    print(f"Saved blended visualization to: {blend_path}")
    
    # Create BDD100K-style JSON structure with depth information
    # Divide the image into a grid for depth samples
    grid_size = 4  # 4x4 grid = 16 regions
    
    # Initialize objects list
    objects = []
    
    # Add depth information for the entire image
    objects.append({
        "category": "depth_full_image",
        "id": 0,
        "attributes": {
            "min_depth": float(min_depth),
            "max_depth": float(max_depth),
            "avg_depth": float(avg_depth)
        }
    })
    
    # Create a grid of areas with average depth
    # Each entry will be a region with a polygon outline and average depth
    grid_x = np.linspace(0, img_width, grid_size + 1, dtype=int)
    grid_y = np.linspace(0, img_height, grid_size + 1, dtype=int)
    
    obj_id = 1
    for i in range(grid_size):
        for j in range(grid_size):
            x1, x2 = grid_x[j], grid_x[j+1]
            y1, y2 = grid_y[i], grid_y[i+1]
            
            # Create polygon for this grid cell
            poly = [
                [float(x1), float(y1), "L"],
                [float(x2), float(y1), "L"],
                [float(x2), float(y2), "L"],
                [float(x1), float(y2), "L"],
                [float(x1), float(y1), "L"],  # Close the polygon
            ]
            
            # Calculate average depth for this region
            region_depth = depth_map[y1:y2, x1:x2].mean()
            
            # Add this region as an object
            objects.append({
                "category": "depth_region",
                "id": obj_id,
                "attributes": {
                    "region": f"grid_{i}_{j}",
                    "depth": float(region_depth)
                },
                "poly2d": poly
            })
            obj_id += 1
    
    # Create final JSON structure
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
            "timeofday": "dawn/dusk",
            "depth_stats": {
                "min": float(min_depth),
                "max": float(max_depth),
                "average": float(avg_depth)
            }
        }
    }
    
    # Save JSON
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{image_name}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved depth json to: {json_path}")

def main():
    # Create file dialog
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select image for depth estimation",
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
    print("\nDepth estimation complete!")

if __name__ == "__main__":
    main() 