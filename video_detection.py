import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from tkinter import Tk, filedialog

def process_video(video_path, model_path, output_dir='output_frames', frame_skip=2):
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Create video-specific output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Information:")
    print(f"FPS: {fps}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Processing every {frame_skip} frames")
    print(f"Expected output frames: {total_frames//frame_skip}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every nth frame
        if frame_count % frame_skip == 0:
            # Perform detection
            results = model(frame)
            
            # Get the first result (since we're processing one frame at a time)
            result = results[0]
            
            # Draw bounding boxes and confidence scores
            annotated_frame = result.plot()
            
            # Save the frame with detections
            output_path = os.path.join(video_output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(output_path, annotated_frame)
            
            # Print detection information
            print(f"\nFrame {frame_count} (Saved as {saved_count}):")
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                print(f"Detected {class_name} with confidence: {confidence:.2f}")
            
            saved_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"\nProcessing complete.")
    print(f"Processed {frame_count} frames")
    print(f"Saved {saved_count} frames")
    print(f"Output frames saved in: {video_output_dir}")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    
    if not video_path:  
        print("No video file selected. Exiting...")
        exit()
    
    # Ask for frame skip value
    try:
        frame_skip = int(input("Enter frame skip value (e.g., 2 for every 2nd frame, 3 for every 3rd frame): "))
        if frame_skip < 1:
            print("Frame skip must be at least 1. Using default value of 2.")
            frame_skip = 2
    except ValueError:
        print("Invalid input. Using default frame skip value of 2.")
        frame_skip = 2
        
    model_path = "runs/detect/train3/weights/best.pt"
    process_video(video_path, model_path, frame_skip=frame_skip) 