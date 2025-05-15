import os
import json
import torch
from torchvision.io import read_image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class_names = ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]
class_to_id = {name: idx for idx, name in enumerate(class_names)}

splits = ["train", "val", "test"]
image_dirs = {split: os.path.join("bdd_images", split) for split in splits}
label_dirs = {split: os.path.join("bdd_labels", split) for split in splits}
output_label_dirs = {split: os.path.join("bdd_images", split) for split in splits}

for split in splits:
    os.makedirs(output_label_dirs[split], exist_ok=True)

def convert_box2d_to_yolo(box2d, img_width, img_height):
    x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def process_image(json_file, split, image_dir, label_dir, output_label_dir):
    if not json_file.endswith(".json"):
        return
    json_path = os.path.join(label_dir, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data.get("name")
    image_path = os.path.join(image_dir, image_name + ".jpg")

    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping...")
        return

    try:
        img = read_image(image_path).to(device)
        img_height, img_width = img.shape[1], img.shape[2]
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return

    yolo_lines = []
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            category = obj.get("category")
            if category not in class_names:
                continue
            box2d = obj.get("box2d")
            if not box2d:
                continue
            class_id = class_to_id[category]
            x_center, y_center, width, height = convert_box2d_to_yolo(box2d, img_width, img_height)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    output_txt = os.path.join(output_label_dir, json_file.replace(".json", ".txt"))
    with open(output_txt, "w") as f:
        f.write("\n".join(yolo_lines))

def process_split(split):
    print(f"Processing {split} split...")
    image_dir = image_dirs[split]
    label_dir = label_dirs[split]
    output_label_dir = output_label_dirs[split]

    json_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]

    num_workers = min(mp.cpu_count(), 8)
    with mp.Pool(num_workers) as pool:
        pool.map(
            partial(
                process_image,
                split=split,
                image_dir=image_dir,
                label_dir=label_dir,
                output_label_dir=output_label_dir
            ),
            json_files
        )

if __name__ == "__main__":
    for split in splits:
        process_split(split)
    print("Conversion complete!")
