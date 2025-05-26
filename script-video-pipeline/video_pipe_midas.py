import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import shutil

class_names = [
    "person", "bike", "car", "motor", "bus", "train", "truck", "traffic light", "traffic sign", "rider"
]

def create_mask_from_polygon(polygon, img_shape):
    """Create a binary mask from polygon coordinates"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if not polygon:
        return mask
    points = np.array([[int(p[0]), int(p[1])] for p in polygon if p[2] == "L"], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask

def compute_masked_depth(depth_map, mask):
    """Compute depth statistics for a masked region"""
    masked_depth = depth_map[mask > 0]
    if len(masked_depth) == 0:
        return None
    return {
        "min": float(np.min(masked_depth)),
        "max": float(np.max(masked_depth)),
        "mean": float(np.mean(masked_depth)),
        "std": float(np.std(masked_depth))
    }

def create_depth_visualization(depth_map, mask=None):
    """Create a color visualization of the depth map"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if mask is not None:
        depth_colored[mask > 0] = [0, 255, 0]  # Green overlay for masked regions
    return depth_colored

def process_frames(input_dir, output_dir, seg_json_path):
    """Process all frames in input_dir using MiDaS, calculate depth for each segmented object, and output a JSON file with per-frame predictions."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
    model.eval()

    # Load segmentation JSON
    with open(seg_json_path, 'r') as f:
        seg_data = json.load(f)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    if not frame_files:
        print("Error: No frame files found in the selected directory")
        return False

    aggregated_data = {
        "name": os.path.basename(input_dir),
        "frames": [],
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "dawn/dusk"
        }
    }

    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to load image: {frame_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_height, img_width = img.shape[:2]

        # Get depth map for the whole frame
        with torch.no_grad():
            inputs = processor(images=img_pil, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(img_height, img_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

        # Get segmented objects for this frame
        seg_objects = seg_data["frames"][frame_idx]["objects"]
        objects = []

        # Create visualization
        vis_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Get all depths for color mapping
        depths = []
        for seg_obj in seg_objects:
            polygon = seg_obj["poly2d"]
            mask = create_mask_from_polygon(polygon, img.shape)
            depth_stats = compute_masked_depth(depth_map, mask)
            if depth_stats:
                depths.append(depth_stats["mean"])

        if depths:
            # Convert depths to inverse scale for color mapping (100/depth)
            inverse_depths = [100/d for d in depths]
            min_depth = min(inverse_depths)
            max_depth = max(inverse_depths)
            
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=min_depth, vmax=max_depth)
            
            for seg_obj in seg_objects:
                # Get polygon and create mask
                polygon = seg_obj["poly2d"]
                mask = create_mask_from_polygon(polygon, img.shape)
                
                # Compute depth statistics for this object
                depth_stats = compute_masked_depth(depth_map, mask)
                
                if depth_stats:
                    # Calculate color based on depth
                    depth = 100/depth_stats["mean"]  # Convert to inverse scale for coloring
                    color = cmap(norm(depth))[:3]
                    color = tuple(min(255, int(c * 255 * 1.5)) for c in color)  # Make color lighter
                    
                    # Add to visualization
                    vis_mask[mask > 0] = color
                    
                    # Calculate center point for caption
                    points = np.array([[int(p[0]), int(p[1])] for p in polygon if p[2] == "L"], dtype=np.int32)
                    if len(points) > 0:
                        M = cv2.moments(points)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            
                            # Add caption
                            text = f"{seg_obj['category']}: {depth_stats['mean']:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                            
                            # Draw background rectangle for text
                            cv2.rectangle(vis_mask, 
                                        (center_x - text_width//2 - 5, center_y - text_height - 5),
                                        (center_x + text_width//2 + 5, center_y + 5),
                                        (0, 0, 0), -1)
                            
                            # Draw text
                            cv2.putText(vis_mask, text,
                                      (center_x - text_width//2, center_y),
                                      font, font_scale, (255, 255, 255), thickness)
                    
                    # Create object with both segmentation and depth info
                    obj = {
                        "category": seg_obj["category"],
                        "id": seg_obj["id"],
                        "attributes": seg_obj["attributes"],
                        "poly2d": polygon,
                        "depth": depth_stats,
                        "confidence": seg_obj.get("confidence", None)
                    }
                    objects.append(obj)

            # Blend visualization with original image
            alpha = 0.6
            blended = cv2.addWeighted(img, 1 - alpha, vis_mask, alpha, 0)
        else:
            blended = img.copy()

        # Save visualization
        out_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(out_frame_path, blended)

        frame_data = {
            "timestamp": seg_data["frames"][frame_idx]["timestamp"],  
            "objects": objects
        }
        aggregated_data["frames"].append(frame_data)

    # Save MIDAS JSON in output_dir directly
    temp_json_path = os.path.join(output_dir, 'midas_output.json')
    with open(temp_json_path, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    # Copy to outputs_vid/midas/json with new name
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_json_dir = os.path.join(project_root, 'outputs_vid', 'midas', 'json')
    os.makedirs(out_json_dir, exist_ok=True)
    video_name = os.path.basename(os.path.normpath(input_dir))
    out_json_name = f"{video_name}.json"
    out_json_path = os.path.join(out_json_dir, out_json_name)
    shutil.copy(temp_json_path, out_json_path)
    print(f"Saved MiDaS JSON to: {out_json_path}")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        seg_json_path = sys.argv[3]
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory.")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
        process_frames(input_dir, output_dir, seg_json_path)
    else:
        print("Usage: python video_pipe_midas.py <input_dir> <output_dir> <seg_json_path>")
        sys.exit(1)