import cv2
import os
from tkinter import Tk, filedialog

def reconstruct_video(frames_dir):
    frame_files = [f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')]
    frame_files.sort()
    
    if not frame_files:
        print("Error: No frame files found in the selected directory")
        return
    
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print("Error: Could not read first frame")
        return
    
    height, width = first_frame.shape[:2]

    output_video_path = os.path.join(frames_dir, 'reconstructed_video.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    
    total_frames = len(frame_files)
    for i, frame_file in enumerate(frame_files, 1):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Error: Could not read frame {frame_file}")
            continue
        
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        print(f"Processing frame {i}/{total_frames}", end='\r')
    
    out.release()
    print(f"\nVideo reconstruction complete. Output saved to: {output_video_path}")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    
    frames_dir = filedialog.askdirectory(
        title="Select Directory Containing Processed Frames"
    )
    
    if not frames_dir:
        print("No directory selected. Exiting...")
        exit()
    
    reconstruct_video(frames_dir) 