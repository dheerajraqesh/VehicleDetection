import os
import json
import cv2
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
import shutil

# --- CONFIG ---
class_names = [
    "person", "bike", "car", "motor", "bus", "train", "truck", "traffic light", "traffic sign", "rider"
]
class_colors = {
    "person": (255, 0, 0),        # Blue
    "bike": (0, 255, 0),          # Green
    "car": (0, 0, 255),           # Red
    "motor": (255, 255, 0),       # Cyan
    "bus": (255, 0, 255),         # Magenta
    "train": (0, 255, 255),       # Yellow
    "truck": (128, 0, 128),       # Purple
    "traffic light": (0, 128, 255), # Orange
    "traffic sign": (128, 255, 0),  # Light Green
    "rider": (200, 200, 200),     # Gray
}

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates in BDD format"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    polygon = []
    for point in approx:
        x, y = point[0]
        polygon.append([float(x), float(y), "L"])
    if polygon:
        polygon.append([float(polygon[0][0]), float(polygon[0][1]), "L"])
    return polygon

def process_frames(input_dir, output_dir, yolo_json_path):
    """Process all frames in input_dir using Mask2Former on YOLO bboxes, save annotated frames to output_dir, and output a JSON file with per-frame predictions."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance").to(device)
    model.eval()

    # Load YOLO JSON
    with open(yolo_json_path, 'r') as f:
        yolo_data = json.load(f)

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

    # Process each frame
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to load image: {frame_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_height, img_width = img.shape[:2]

        # Get YOLO detections for this frame
        yolo_objects = yolo_data["frames"][frame_idx]["objects"]
        objects = []

        # Process each object in the frame
        for obj_id, yolo_obj in enumerate(yolo_objects):
            category = yolo_obj["category"]
            if category not in class_names:
                continue

            # Get bbox coordinates
            bbox = yolo_obj["box2d"]
            x1, y1, x2, y2 = map(int, [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            
            # Crop image to bbox
            roi = img_pil.crop((x1, y1, x2, y2))
            roi_width, roi_height = roi.size

            # Run segmentation on cropped region
            with torch.no_grad():
                inputs = processor(images=roi, return_tensors="pt").to(device)
                outputs = model(**inputs)
                target_sizes = [(roi_height, roi_width)]
                results = processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)[0]
                segmentation = results["segmentation"].cpu().numpy()
                segments_info = results["segments_info"]

                # Find best mask (largest area)
                best_mask = None
                best_area = 0
                for seg in segments_info:
                    mask = (segmentation == seg["id"]).astype(np.uint8)
                    area = mask.sum()
                    if area > best_area:
                        best_area = area
                        best_mask = mask

                if best_mask is not None:
                    # Resize mask to original bbox size
                    mask_resized = cv2.resize(best_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2][mask_resized > 0] = 1

                    # Convert mask to polygon
                    polygon = mask_to_polygon(full_mask)
                    if polygon:
                        # Get color for visualization (lighter version)
                        base_color = class_colors.get(category, (255, 255, 255))
                        color = tuple(min(255, int(c * 1.5)) for c in base_color)  # Make color lighter
                        
                        # Create overlay for the mask
                        overlay = img.copy()
                        overlay[full_mask > 0] = color
                        # Blend the overlay with the original image
                        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

                        # Calculate center point for caption
                        points = np.array([[int(p[0]), int(p[1])] for p in polygon if p[2] == "L"], dtype=np.int32)
                        center_x = int(np.mean(points[:, 0]))
                        center_y = int(np.mean(points[:, 1]))

                        # Add caption
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(category, font, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(img, 
                                    (center_x - text_width//2 - 5, center_y - text_height - 5),
                                    (center_x + text_width//2 + 5, center_y + 5),
                                    (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(img, category,
                                  (center_x - text_width//2, center_y),
                                  font, font_scale, (255, 255, 255), thickness)

                        # Create object data
                        obj = {
                            "category": category,
                            "id": obj_id,
                            "attributes": yolo_obj["attributes"],
                            "poly2d": polygon,
                            "confidence": yolo_obj.get("confidence", None)
                        }
                        objects.append(obj)

        # Save processed frame
        out_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(out_frame_path, img)

        # Add frame data to aggregated output
        frame_data = {
            "timestamp": yolo_data["frames"][frame_idx]["timestamp"],  # Use timestamp from YOLO data
            "objects": objects
        }
        aggregated_data["frames"].append(frame_data)

    # Save SEG JSON in output_dir directly
    temp_json_path = os.path.join(output_dir, 'seg_output.json')
    with open(temp_json_path, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        yolo_json_path = sys.argv[3]
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory.")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
        process_frames(input_dir, output_dir, yolo_json_path)
    else:
        print("Usage: python video_pipe_seg.py <input_dir> <output_dir> <yolo_json_path>")
        sys.exit(1)
