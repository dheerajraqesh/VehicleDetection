import os
import json
import cv2
import torch
import numpy as np
import argparse
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt

# Define classes to match the pipeline
class_names = [
    "person", "bike", "car", "motor", "bus", "train", "truck", "traffic light", "traffic sign", "rider"
]

def create_mask_from_polygon(polygon, image_shape):
    """Convert polygon coordinates to binary mask"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array([[int(p[0]), int(p[1])] for p in polygon if p[2] in ["L", "C"]])
    if len(points) > 2:  # Need at least 3 points for a polygon
        cv2.fillPoly(mask, [points], 1)
    return mask

def compute_masked_depth(depth_map, mask):
    """Compute average depth for a masked region"""
    if mask.sum() == 0:
        return None
    masked_depth = depth_map * mask
    average_depth = masked_depth.sum() / mask.sum()
    return float(average_depth)

def create_depth_visualization(img, objects_with_depth, image_name):
    """Create visualization of objects colored by depth with text labels"""
    # Create output directories
    vis_dir = "outputs/midas/img"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a copy of the original image
    vis_img = img.copy()
    h, w = img.shape[:2]
    
    # Create a mask for the visualization
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get all depths for normalization
    depths = [obj['depth'] for obj in objects_with_depth if 'depth' in obj]
    if not depths:
        return
    
    min_depth = min(depths)
    max_depth = max(depths)
    depth_range = max_depth - min_depth if max_depth > min_depth else 1.0
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Keep track of object centers and info for labeling
    object_info = []
    
    # Draw each object with color based on depth
    for obj in objects_with_depth:
        if 'depth' not in obj:
            continue
            
        depth = obj['depth']
        category = obj.get('category', 'unknown')
        
        # Skip background for text labels
        if category == 'background':
            continue
            
        # Normalize depth and get color
        norm_depth = (depth - min_depth) / depth_range
        color = cmap(norm_depth)[:3]  # Get RGB, exclude alpha
        color = tuple(int(c * 255) for c in color)  # Convert to 0-255 range
        
        # Create a mask for this object
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        
        center_x, center_y = None, None
        
        if 'poly2d' in obj:
            # Create mask from polygon
            obj_mask = create_mask_from_polygon(obj['poly2d'], img.shape)
            
            # Calculate centroid for text placement
            points = np.array([[int(p[0]), int(p[1])] for p in obj['poly2d'] if p[2] in ["L", "C"]])
            if len(points) > 0:
                M = cv2.moments(points)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
        
        elif 'box2d' in obj:
            # Handle bounding box objects
            box = [
                int(obj['box2d']['x1']), int(obj['box2d']['y1']),
                int(obj['box2d']['x2']), int(obj['box2d']['y2'])
            ]
            cv2.rectangle(obj_mask, (box[0], box[1]), (box[2], box[3]), 1, -1)
            
            # Use box center for text placement
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
        
        # Apply color to the mask
        vis_mask[obj_mask > 0] = color
        
        # Store object info for text labels if we found a center
        if center_x is not None and center_y is not None:
            # Calculate a good text position
            object_info.append({
                'center': (center_x, center_y),
                'category': category,
                'depth': depth,
                'color': color
            })
    
    # Blend the visualization mask with the original image
    alpha = 0.6  # Transparency factor
    blended = cv2.addWeighted(img, 1 - alpha, vis_mask, alpha, 0)
    
    # Add text for each object
    for info in object_info:
        text = f"{info['category']}: {info['depth']:.2f}"
        center_x, center_y = info['center']
        
        # Determine text color (white or black) based on background
        color_sum = sum(info['color'])
        text_color = (0, 0, 0) if color_sum > 380 else (255, 255, 255)
        
        # Add text with background box for better visibility
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        
        # Draw background rectangle for text
        cv2.rectangle(blended, 
                     (center_x - text_w//2 - 2, center_y - text_h - 5),
                     (center_x + text_w//2 + 2, center_y + 5), 
                     info['color'], -1)
        
        # Draw text
        cv2.putText(blended, text, 
                   (center_x - text_w//2, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Create figure for the visualization with colorbar only
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    
    # Add colorbar for depth range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_depth, max_depth))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Depth')
    
    plt.title(f'Depth Visualization - {image_name}')
    plt.tight_layout()
    
    # Save the visualization
    vis_path = os.path.join(vis_dir, f"{image_name}.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save a direct OpenCV version for simpler viewing
    cv_vis_path = os.path.join(vis_dir, f"{image_name}_simple.png")
    cv2.imwrite(cv_vis_path, blended)
    
    print(f"Saved depth visualization to: {vis_path}")
    return blended

def process_image(image_path, json_path):
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    midas_json_dir = "outputs/midas/json"
    os.makedirs(midas_json_dir, exist_ok=True)

    # Load pre-trained MiDaS model
    print("Loading MiDaS model...")
    processor = AutoImageProcessor.from_pretrained("Intel/dpt-large", use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.to(device)
    model.eval()

    # Load JSON and image
    print(f"Processing: {image_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data["name"]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    # Prepare image for model
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

    # Initialize output data structure
    output_data = {
        "name": image_name,
        "frames": [{
            "timestamp": 10000,
            "objects": []
        }],
        "attributes": data.get("attributes", {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "dawn/dusk"
        })
    }

    # Create a mask for processed areas
    processed_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Process objects and compute depths
    all_objects_with_depth = []
    
    for frame in data["frames"]:
        for obj in frame["objects"]:
            if "poly2d" in obj:
                # Create mask from polygon
                mask = create_mask_from_polygon(obj["poly2d"], img.shape)
                depth = compute_masked_depth(depth_map, mask)
                
                if depth is not None:
                    # Add object with depth to output
                    obj_data = obj.copy()
                    obj_data["depth"] = depth
                    output_data["frames"][0]["objects"].append(obj_data)
                    all_objects_with_depth.append(obj_data)
                    
                    # Update processed mask
                    processed_mask = cv2.bitwise_or(processed_mask, mask)
            
            elif "box2d" in obj and obj["category"] in class_names:
                # Handle bounding box objects
                box = [obj["box2d"]["x1"], obj["box2d"]["y1"], 
                       obj["box2d"]["x2"], obj["box2d"]["y2"]]
                box = [int(x) for x in box]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[box[1]:box[3], box[0]:box[2]] = 1
                depth = compute_masked_depth(depth_map, mask)
                
                if depth is not None:
                    obj_data = obj.copy()
                    obj_data["depth"] = depth
                    output_data["frames"][0]["objects"].append(obj_data)
                    all_objects_with_depth.append(obj_data)
                    
                    # Update processed mask
                    processed_mask = cv2.bitwise_or(processed_mask, mask)

    # Compute depth for unprocessed areas
    unprocessed_mask = (1 - processed_mask).astype(bool)
    if unprocessed_mask.any():
        background_depth = compute_masked_depth(depth_map, unprocessed_mask)
        if background_depth is not None:
            # Add background depth as a special object
            background_obj = {
                "category": "background",
                "id": len(output_data["frames"][0]["objects"]),
                "depth": background_depth,
                "attributes": {
                    "is_background": True
                }
            }
            output_data["frames"][0]["objects"].append(background_obj)
            all_objects_with_depth.append(background_obj)

    # Save depths JSON
    output_path = os.path.join(midas_json_dir, f"{image_name}.json")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved depths JSON to: {output_path}")
    
    # Create and save depth visualization
    create_depth_visualization(img, all_objects_with_depth, image_name)

def main():
    parser = argparse.ArgumentParser(description="Generate depth information using MiDaS")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--json", required=True, help="Path to segmentation JSON")
    args = parser.parse_args()

    process_image(args.image, args.json)
    print("\nDepth estimation complete!")

if __name__ == "__main__":
    main()