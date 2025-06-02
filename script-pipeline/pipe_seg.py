import os
import json
import cv2
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np

base_path = r"E:\Vehicle Occlusion"
colored_dir = os.path.join(base_path, "outputs/seg/img")
json_dir = os.path.join(base_path, "outputs/seg/json")
os.makedirs(colored_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

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

def process_image(json_path, img_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_height, img_width = img.shape[:2]
    objects = data["frames"][0]["objects"]
    image_name = data["name"]

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

    # --- MODEL ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance").to(device)
    model.eval()

    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for obj_id, obj in enumerate(objects):
        category = obj["category"]
        if category not in class_names:
            continue

        box2d = obj["box2d"]
        x1, y1, x2, y2 = map(int, [box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]])
        roi = img_pil.crop((x1, y1, x2, y2))
        roi_width, roi_height = roi.size

        with torch.no_grad():
            inputs = processor(images=roi, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = [(roi_height, roi_width)]
            results = processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)[0]
            segmentation = results["segmentation"].cpu().numpy()
            segments_info = results["segments_info"]

            best_mask = None
            best_area = 0
            for seg in segments_info:
                mask = (segmentation == seg["id"]).astype(np.uint8)
                area = mask.sum()
                if area > best_area:
                    best_area = area
                    best_mask = mask

            if best_mask is not None:
                mask_resized = cv2.resize(best_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                full_mask[y1:y2, x1:x2][mask_resized > 0] = 1

                polygon = mask_to_polygon(full_mask)
                if polygon:
                    color = class_colors.get(category, (255, 255, 255))
                    colored_mask[full_mask > 0] = color

                    obj_data = {
                        "category": category,
                        "id": obj_id,
                        "attributes": obj["attributes"],
                        "poly2d": polygon
                    }
                    output_data["frames"][0]["objects"].append(obj_data)

    colored_path = os.path.join(colored_dir, f"{image_name}.png")
    cv2.imwrite(colored_path, colored_mask)

    json_path = os.path.join(json_dir, f"{image_name}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved colored mask to {colored_path}")
    print(f"Saved polygon JSON to {json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mask2Former direct class overlay pipeline")
    parser.add_argument('--json', required=True, help='Path to YOLO JSON file')
    parser.add_argument('--image', required=True, help='Path to input image file')
    args = parser.parse_args()
    process_image(args.json, args.image)
