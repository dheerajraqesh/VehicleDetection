import os
import json
import cv2
import torch
import numpy as np
import argparse
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class_names = [
    "person", "bike", "car", "motor", "bus", "train", "truck", "traffic light", "traffic sign", "rider"
]

def create_mask_from_polygon(polygon, image_shape):
    """Convert polygon coordinates to binary mask"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array([[int(p[0]), int(p[1])] for p in polygon if p[2] in ["L", "C"]])
    if len(points) > 2:  
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
    vis_dir = "outputs/midas/img"
    os.makedirs(vis_dir, exist_ok=True)
    vis_img = img.copy()
    h, w = img.shape[:2]
    
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    depths = [obj['depth'] for obj in objects_with_depth if 'depth' in obj]
    if not depths:
        return
    
    min_depth = min(depths)
    max_depth = max(depths)
    depth_range = max_depth - min_depth if max_depth > min_depth else 1.0
    
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=min_depth, vmax=max_depth)
    
    object_info = []
    
    for obj in objects_with_depth:
        if 'depth' not in obj:
            continue
            
        depth = obj['depth']
        category = obj.get('category', 'unknown')
        
        if category == 'background':
            continue
            
        color = cmap(norm(depth))[:3] 
        color = tuple(int(c * 255) for c in color) 
        
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        
        center_x, center_y = None, None
        
        if 'poly2d' in obj:
            obj_mask = create_mask_from_polygon(obj['poly2d'], img.shape)
            
            points = np.array([[int(p[0]), int(p[1])] for p in obj['poly2d'] if p[2] in ["L", "C"]])
            if len(points) > 0:
                M = cv2.moments(points)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
        
        elif 'box2d' in obj:
            box = [
                int(obj['box2d']['x1']), int(obj['box2d']['y1']),
                int(obj['box2d']['x2']), int(obj['box2d']['y2'])
            ]
            cv2.rectangle(obj_mask, (box[0], box[1]), (box[2], box[3]), 1, -1)
            
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
        
        vis_mask[obj_mask > 0] = color
        
        if center_x is not None and center_y is not None:
            object_info.append({
                'center': (center_x, center_y),
                'category': category,
                'depth': depth,
                'color': color
            })
    
    alpha = 0.6 
    blended = cv2.addWeighted(img, 1 - alpha, vis_mask, alpha, 0)
    
    for info in object_info:
        text = f"{info['category']}: {info['depth']:.2f}"
        center_x, center_y = info['center']
        
        color_sum = sum(info['color'])
        text_color = (0, 0, 0) if color_sum > 380 else (255, 255, 255)
        
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        
        cv2.rectangle(blended, 
                     (center_x - text_w//2 - 2, center_y - text_h - 5),
                     (center_x + text_w//2 + 2, center_y + 5), 
                     info['color'], -1)
        
        cv2.putText(blended, text, 
                   (center_x - text_w//2, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Depth')
    
    plt.title(f'Depth Visualization - {image_name}')
    plt.tight_layout()
    
    vis_path = os.path.join(vis_dir, f"{image_name}.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved depth visualization to: {vis_path}")
    return blended

def process_image(image_path, json_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    midas_json_dir = "outputs/midas/json"
    os.makedirs(midas_json_dir, exist_ok=True)

    print("Loading MiDaS model...")
    processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.to(device)
    model.eval()

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

    processed_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    all_objects_with_depth = []
    
    for frame in data["frames"]:
        for obj in frame["objects"]:
            if "poly2d" in obj:
                mask = create_mask_from_polygon(obj["poly2d"], img.shape)
                depth = compute_masked_depth(depth_map, mask)
                
                if depth is not None:
                    obj_data = obj.copy()
                    obj_data["depth"] = depth
                    output_data["frames"][0]["objects"].append(obj_data)
                    all_objects_with_depth.append(obj_data)
                    
                    processed_mask = cv2.bitwise_or(processed_mask, mask)
            
            elif "box2d" in obj and obj["category"] in class_names:
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
                    
                    processed_mask = cv2.bitwise_or(processed_mask, mask)

    # Compute depth for unprocessed areas
    unprocessed_mask = (1 - processed_mask).astype(bool)
    if unprocessed_mask.any():
        background_depth = compute_masked_depth(depth_map, unprocessed_mask)
        if background_depth is not None:
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

    output_path = os.path.join(midas_json_dir, f"{image_name}.json")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved depths JSON to: {output_path}")
    
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