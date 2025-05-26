# Video Pipeline Scripts

The `scripts-video-pipeline` folder contains scripts for running the three main models (YOLO, MiDaS, Segmentation) on video input. The pipeline:
- Accepts a video file as input
- Splits the video into frames
- Runs each model on the frames
- Aggregates results into a single output video and a single JSON file (using BDD format with frames notation)

This enables batch processing and tracking of predictions across video frames. (Kalman filtering and tracking will be added later.)
