import os
import tkinter as tk
from tkinter import filedialog

def clear_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select a folder to clear all files")
    if not folder:
        print("No folder selected. Exiting.")
    elif not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
    else:
        clear_files_in_folder(folder)
