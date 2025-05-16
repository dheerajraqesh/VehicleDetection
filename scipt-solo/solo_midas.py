import os
import json
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
    
    if points:
        points.append([float(points[0][0]), float(points[0][1]), "L"])
    
    return points

def add_depth_colorbar(img, min_depth, max_depth, cmap_name='viridis'):
    """Add a depth colorbar to the right side of the image"""
    h, w = img.shape[:2]
    
    bar_width = 30
    padding = 10
    
    expanded_img = np.zeros((h, w + bar_width + padding * 2, 3), dtype=np.uint8)
    expanded_img[:, :w] = img
    
    depth_range = np.linspace(min_depth, max_depth, h)[:, np.newaxis]
    
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=min_depth, vmax=max_depth)
    
    colors = cmap(norm(depth_range))[:, 0, :3]
    
    colors_rgb = (colors * 255).astype(np.uint8)
    
    for i in range(h):
        expanded_img[i, w + padding:w + padding + bar_width] = colors_rgb[i]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    tick_count = 6
    for i in range(tick_count):
        y_pos = int((h - 1) * i / (tick_count - 1))
        depth_val = max_depth - (max_depth - min_depth) * i / (tick_count - 1)
        
        expanded_img[y_pos, w + padding + bar_width:w + padding + bar_width + 5] = [255, 255, 255]
        
        text = f"{depth_val:.1f}"
        text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
        cv2.putText(
            expanded_img, 
            text, 
            (w + padding + bar_width + 8, y_pos + text_size[1]//2), 
            font, 
            font_scale, 
            (255, 255, 255), 
            1
        )
    
    cv2.putText(
        expanded_img,
        "Depth",
        (w + padding, 20),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    return expanded_img

def process_image(image_path):
    """Process a single image with MiDaS depth estimation"""
    print("\nInitializing MiDaS model...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading MiDaS model...")
    processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.to(device)
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    print("Running depth estimation...")
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)

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

    min_depth = depth_map.min()
    max_depth = depth_map.max()
    avg_depth = depth_map.mean()
    
    norm = Normalize(vmin=min_depth, vmax=max_depth)
    depth_norm = norm(depth_map)
    
    cmap = plt.get_cmap('viridis')
    colored_depth = cmap(depth_norm)[:, :, :3] 
    colored_depth = (colored_depth * 255).astype(np.uint8)
    
    info_img = colored_depth.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info_img, f"Min depth: {min_depth:.2f}", (20, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, f"Max depth: {max_depth:.2f}", (20, 60), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, f"Avg depth: {avg_depth:.2f}", (20, 90), font, 0.7, (255, 255, 255), 2)
    
    info_img_with_bar = add_depth_colorbar(info_img, min_depth, max_depth)
    
    vis_path = os.path.join(OUTPUT_IMG_DIR, f"{image_name}.png")
    cv2.imwrite(vis_path, cv2.cvtColor(info_img_with_bar, cv2.COLOR_RGB2BGR))
    print(f"Saved depth visualization to: {vis_path}")
    
    alpha = 0.7
    blend = cv2.addWeighted(img, 1-alpha, cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR), alpha, 0)
    blend_path = os.path.join(OUTPUT_IMG_DIR, f"{image_name}_blend.png")
    cv2.imwrite(blend_path, blend)
    print(f"Saved blended visualization to: {blend_path}")
    
    grid_size = 4 
    
    objects = []
    
    objects.append({
        "category": "depth_full_image",
        "id": 0,
        "attributes": {
            "min_depth": float(min_depth),
            "max_depth": float(max_depth),
            "avg_depth": float(avg_depth)
        }
    })
    
    grid_x = np.linspace(0, img_width, grid_size + 1, dtype=int)
    grid_y = np.linspace(0, img_height, grid_size + 1, dtype=int)
    
    obj_id = 1
    for i in range(grid_size):
        for j in range(grid_size):
            x1, x2 = grid_x[j], grid_x[j+1]
            y1, y2 = grid_y[i], grid_y[i+1]
            
            poly = [
                [float(x1), float(y1), "L"],
                [float(x2), float(y1), "L"],
                [float(x2), float(y2), "L"],
                [float(x1), float(y2), "L"],
                [float(x1), float(y1), "L"],
            ]
            
            region_depth = depth_map[y1:y2, x1:x2].mean()
            
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
    
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{image_name}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved depth json to: {json_path}")

def main():
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
    
    process_image(image_path)
    print("\nDepth estimation complete!")

if __name__ == "__main__":
    main() 