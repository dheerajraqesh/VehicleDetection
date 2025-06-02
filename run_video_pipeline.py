import os
import json
import cv2
import subprocess
import shutil
import sys
import argparse
import tkinter as tk
from tkinter import filedialog

VID_INPUT_DIR = "Videos"
OUTPUTS_VID_DIR = "outputs_vid"
TEMP_DIR = "temp"

os.makedirs(VID_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUTS_VID_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

for model in ['yolo', 'seg', 'midas', 'deepsort']:
    os.makedirs(os.path.join(OUTPUTS_VID_DIR, model, 'vid'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_VID_DIR, model, 'json'), exist_ok=True)

def copy_model_json_to_output(model, src_dir, dst_dir, vidname):
    """Copy only the main output JSON of the model from src_dir to dst_dir, renaming it to vidname.json."""
    import shutil, os
    os.makedirs(dst_dir, exist_ok=True)
    model_json_map = {
        'yolo': 'yolo_output.json',
        'seg': 'seg_output.json',
        'midas': 'midas_output.json',
        'deepsort': 'deepsort_output.json',
    }
    main_json = model_json_map.get(model)
    if main_json:
        src_json = os.path.join(src_dir, main_json)
        dst_json = os.path.join(dst_dir, f"{vidname}.json")
        if os.path.exists(src_json):
            shutil.copy2(src_json, dst_json)

def extract_frames(video_path, output_dir):
    """Extract frames from video to output_dir."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")
    return True

def reconstruct_video(frames_dir, output_video_path, fps=30):
    """Reconstruct video from frames in frames_dir and save to output_video_path."""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    if not frame_files:
        print("Error: No frame files found in the selected directory")
        return False
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print("Error: Could not read first frame")
        return False
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not read frame {frame_file}")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
    out.release()
    print(f"Video reconstruction complete. Output saved to: {output_video_path}")
    return True

def run_model_script(script_path, *args):
    """Run a model script with given arguments."""
    cmd = [sys.executable, script_path] + list(args)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}. Exited with code {result.returncode}")
        return False
    return True

def aggregate_json_files(json_dir, output_json_path):
    """Aggregate JSON files from json_dir into a single BDD100K-style JSON with frames key."""
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    if not json_files:
        print("Error: No JSON files found in the selected directory")
        return False
    aggregated_data = {
        "name": os.path.splitext(os.path.basename(output_json_path))[0],
        "frames": [],
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "dawn/dusk"
        }
    }
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            if "frames" in data and data["frames"]:
                aggregated_data["frames"].extend(data["frames"])
    with open(output_json_path, 'w') as f:
        json.dump(aggregated_data, f, indent=4)
    print(f"Aggregated JSON saved to: {output_json_path}")
    return True

# --- MAIN PIPELINE ---

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = os.path.join(TEMP_DIR, video_name)
    input_dir = os.path.join(temp_dir, 'input')
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    output_paths = {
        'yolo': {
            'video': os.path.join(OUTPUTS_VID_DIR, 'yolo', 'vid', f"{video_name}.mp4"),
            'json': os.path.join(OUTPUTS_VID_DIR, 'yolo', 'json', f"{video_name}.json")
        },
        'seg': {
            'video': os.path.join(OUTPUTS_VID_DIR, 'seg', 'vid', f"{video_name}.mp4"),
            'json': os.path.join(OUTPUTS_VID_DIR, 'seg', 'json', f"{video_name}.json")
        },
        'midas': {
            'video': os.path.join(OUTPUTS_VID_DIR, 'midas', 'vid', f"{video_name}.mp4"),
            'json': os.path.join(OUTPUTS_VID_DIR, 'midas', 'json', f"{video_name}.json")
        },
        'deepsort': {
            'video': os.path.join(OUTPUTS_VID_DIR, 'deepsort', 'vid', f"{video_name}.mp4"),
            'json': os.path.join(OUTPUTS_VID_DIR, 'deepsort', 'json', f"{video_name}.json")
        }
    }

    # Step 1: Extract frames to input_dir
    if not extract_frames(video_path, input_dir):
        return False

    def clear_output_dir():
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            if os.path.isfile(fp):
                if os.path.basename(fp) not in ['yolo_output.json', 'seg_output.json', 'midas_output.json', 'deepsort_output.json', 'frame_pred.jpg', 'frame_pred.json']:
                    os.remove(fp)

    # Step 2: Run YOLO
    clear_output_dir()
    print("\n========= Starting YOLO Detection ==========\n")
    if not run_model_script("script-video-pipeline/video_pipe_yolo.py", input_dir, output_dir, str(fps)):
        return False
    if not reconstruct_video(output_dir, output_paths['yolo']['video'], fps):
        return False
    copy_model_json_to_output('yolo', output_dir, os.path.join(OUTPUTS_VID_DIR, 'yolo', 'json'), video_name)

    yolo_json = os.path.join(output_dir, "yolo_output.json")
    clear_output_dir()
    print("\n========= Finished YOLO Detection ==========\n")
    # Step 3: Run Mask2Former using YOLO JSON
    print("\n========= Starting Segmentation ==========\n")
    if not run_model_script("script-video-pipeline/video_pipe_seg.py", input_dir, output_dir, yolo_json):
        return False
    if not reconstruct_video(output_dir, output_paths['seg']['video'], fps):
        return False
    copy_model_json_to_output('seg', output_dir, os.path.join(OUTPUTS_VID_DIR, 'seg', 'json'), video_name)

    seg_json = os.path.join(output_dir, "seg_output.json")
    clear_output_dir()
    print("\n========= Finished Segmentation ==========\n")
    # Step 4: Run MiDaS using segmentation JSON
    print("\n========= Starting Midas Depth Estimation ==========\n")
    seg_json = os.path.join(output_dir, "seg_output.json")
    if not run_model_script("script-video-pipeline/video_pipe_midas.py", input_dir, output_dir, seg_json):
        return False
    if not reconstruct_video(output_dir, output_paths['midas']['video'], fps):
        return False
    copy_model_json_to_output('midas', output_dir, os.path.join(OUTPUTS_VID_DIR, 'midas', 'json'), video_name)

    clear_output_dir()
    print("\n========= Finished Midas Depth Estimation ==========\n")
    # Step 5: Run DeepSORT tracking
    midas_json = os.path.join(output_dir, "midas_output.json")
    print("\n========= Starting DeepSORT Tracking ==========\n")
    if not run_model_script("script-video-pipeline/video_pipe_deepsort.py", input_dir, output_dir, midas_json):
        return False
    if not reconstruct_video(output_dir, output_paths['deepsort']['video'], fps):
        return False
    copy_model_json_to_output('deepsort', output_dir, os.path.join(OUTPUTS_VID_DIR, 'deepsort', 'json'), video_name)
    last_img_src = os.path.join(output_dir, "last_frame.jpg")
    last_img_dst = os.path.join(os.path.dirname(output_paths['deepsort']['video']), f"{video_name}_last.jpg")
    if os.path.exists(last_img_src):
        shutil.copy2(last_img_src, last_img_dst)
    last_json_src = os.path.join(output_dir, "last_frame.json")
    last_json_dst = os.path.join(os.path.dirname(output_paths['deepsort']['json']), f"{video_name}_last.json")
    if os.path.exists(last_json_src):
        shutil.copy2(last_json_src, last_json_dst)
    pred_img_src = os.path.join(output_dir, "deepsort_preds.mp4")
    pred_img_dst = os.path.join(os.path.dirname(output_paths['deepsort']['video']), f"{video_name}_preds.mp4")
    if os.path.exists(pred_img_src):
        shutil.copy2(pred_img_src, pred_img_dst)
    pred_json_src = os.path.join(output_dir, "deepsort_preds.json")
    pred_json_dst = os.path.join(os.path.dirname(output_paths['deepsort']['json']), f"{video_name}_preds.json")
    if os.path.exists(pred_json_src):
        shutil.copy2(pred_json_src, pred_json_dst)
    clear_output_dir()
    print("\n========= Finished DeepSORT Tracking =========\n")
    # Step 6: Clean up temp directory
    # --- Run occlusion detection on DeepSORT output ---
    deepsort_json = os.path.join(output_dir, "deepsort_output.json")
    run_model_script("script-video-pipeline/occlusion_detection.py", deepsort_json)
    t=os.path.join(temp_dir, "input")
    shutil.rmtree(t)
    return True

def main():
    parser = argparse.ArgumentParser(description="Process videos through YOLO, Mask2Former, MiDaS, and DeepSORT pipeline.")
    parser.add_argument("--video", help="Path to a specific video file to process. If not provided, a file browser will open.")
    args = parser.parse_args()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file {args.video} does not exist.")
            return
        process_video(args.video)
    else:
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            initialdir=VID_INPUT_DIR,
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if video_path: 
            print(f"\nProcessing video: {video_path}")
            process_video(video_path)
        else:
            print("No video file selected.")

if __name__ == "__main__":
    main() 