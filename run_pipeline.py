import subprocess
import sys
import os

def run_script(args, description=None):
    if description:
        print(f"\n=== {description} ===")
    print(f"Running: {' '.join([str(a) for a in args])}")
    result = subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"Error running {' '.join([str(a) for a in args])}. Exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"=== Finished {str(args[1])} ===\n")

if __name__ == "__main__":
    # NOTE: Required packages:
    # - Mask2Former: 'transformers', 'timm', 'accelerate'
    # - MiDaS: 'torch', 'opencv-python'
    # pip install transformers timm accelerate torch opencv-python
    import tkinter as tk
    from tkinter import filedialog
    import os
    import sys
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select image to process",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        initialdir="E:/Vehicle Occlusion/bdd_images/test"
    )
    if not image_path:
        print("No image selected. Exiting...")
        sys.exit(0)
    run_script([sys.executable, "script-pipeline/pipe_yolo.py", image_path], description="YOLO to JSON")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join("outputs/yolo/json", f"{image_name}.json")
    if not os.path.exists(json_path):
        print(f"Could not find expected JSON output: {json_path}")
        sys.exit(1)
    run_script([
        sys.executable, "script-pipeline/pipe_seg.py",
        "--json", json_path,
        "--image", image_path
    ], description="Mask2Former Direct Class Overlay")
    seg_json_path = os.path.join("outputs/seg/json", f"{image_name}.json")
    run_script([
        sys.executable, "script-pipeline/pipe_midas.py",
        "--json", seg_json_path,
        "--image", image_path
    ], description="MiDaS Depth Estimation")
    print("\nPipeline completed successfully.")
