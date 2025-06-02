import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import shutil

DEFAULT_ATTRIBUTES = {
    "occluded": False,
    "truncated": False,
    "trafficLightColor": "none"
}

def draw_minimal_bbox(img, x1, y1, x2, y2, class_name, color=(0, 255, 0)):
    """Draw a minimal bounding box with class name"""
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    cv2.putText(img, class_name, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    return img

def process_frames(input_dir, output_dir, fps=30):
    """Process all frames in input_dir using YOLO, save annotated frames back, and output a JSON file with per-frame predictions."""
    model = YOLO('runs/detect/train3/weights/best.pt')
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
        results = model(frame_path)
        result = results[0]
        objects = []
        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            img = draw_minimal_bbox(img, x1, y1, x2, y2, class_name)
            obj = {
                "category": class_name,
                "id": idx,
                "attributes": DEFAULT_ATTRIBUTES.copy(),
                "box2d": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "confidence": float(box.conf[0])
            }
            objects.append(obj)
        out_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(out_frame_path, img)
        # Calculate timestamp based on frame number and FPS
        timestamp = int((frame_idx / fps) * 1000)  # Convert to milliseconds
        frame_data = {
            "timestamp": timestamp,
            "objects": objects
        }
        aggregated_data["frames"].append(frame_data)

    # Save YOLO JSON in temp/output directly
    temp_json_path = os.path.join(output_dir, 'yolo_output.json')
    with open(temp_json_path, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        fps = 30  # Default FPS
        if len(sys.argv) > 3:
            fps = float(sys.argv[3])
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory.")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
        process_frames(input_dir, output_dir, fps)
    else:
        print("Usage: python video_pipe_yolo.py <input_dir> <output_dir> [fps]")
        sys.exit(1)
