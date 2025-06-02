import os
import sys
import json
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def poly2d_to_bbox(poly2d):
    # poly2d: list of [x, y, 'L']
    points = np.array([[p[0], p[1]] for p in poly2d if len(p) >= 2])
    if points.shape[0] == 0:
        return [0, 0, 0, 0]
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]

import cv2

def draw_bbox_and_id(img, bbox, track_id, category, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} {category}"
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def process_frames(input_dir, output_dir, midas_json_path):
    with open(midas_json_path, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    tracker = DeepSort(max_age=30, n_init=1)
    # Build category to class_id mapping
    class_names = [
        "person", "bike", "car", "motor", "bus", "train", "truck", "traffic light", "traffic sign", "rider"
    ]
    all_categories = sorted(class_names)
    category_to_id = {cat: i for i, cat in enumerate(all_categories)}
    last_obj_info = {}
    track_histories = {}
    # Prepare output dirs

    # Get video name
    frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    aggregated_frames = []
    for frame_idx, frame in enumerate(frames):
        objects = frame['objects']
        detections = []
        for obj in objects:
            bbox = poly2d_to_bbox(obj['poly2d'])
            confidence = float(obj.get('confidence', 1.0))
            class_id = category_to_id.get(obj['category'], 0)
            detections.append([[bbox[0], bbox[1], bbox[2], bbox[3]], confidence, class_id])
        img_path = os.path.join(input_dir, frame_files[frame_idx])
        img = cv2.imread(img_path)
        tracks = tracker.update_tracks(detections, frame=img)
        # Assign track_id to objects and record last info
        for obj, track in zip(objects, tracks):
            if track.is_confirmed():
                obj['track_id'] = int(track.track_id)
                centroid = np.mean(np.array([[p[0], p[1]] for p in obj['poly2d']]), axis=0)
                if obj['track_id'] not in track_histories:
                    track_histories[obj['track_id']] = {
                        'centroids': [centroid],
                        'timestamps': [frame['timestamp']],
                        'depths': [obj['depth']['mean'] if 'depth' in obj and isinstance(obj['depth'], dict) and 'mean' in obj['depth'] else 0],
                        'speed_xs': [0],
                        'speed_ys': [0],
                        'speeds': [0]
                    }
                    obj['speed_x'] = 0
                    obj['speed_y'] = 0
                    obj['speed'] = 0
                else:
                    last_centroid = track_histories[obj['track_id']]['centroids'][-1]
                    last_timestamp = track_histories[obj['track_id']]['timestamps'][-1]
                    dt = frame['timestamp'] - last_timestamp
                    if dt > 0:
                        raw_speed_x = (centroid[0] - last_centroid[0]) / dt
                        raw_speed_y = (centroid[1] - last_centroid[1]) / dt
                        alpha = 0.8
                        prev_speed_x = track_histories[obj['track_id']]['speed_xs'][-1]
                        prev_speed_y = track_histories[obj['track_id']]['speed_ys'][-1]
                        speed_x = alpha * prev_speed_x + (1 - alpha) * raw_speed_x
                        speed_y = alpha * prev_speed_y + (1 - alpha) * raw_speed_y
                        speed = np.sqrt(speed_x**2 + speed_y**2)
                    else:
                        speed_x, speed_y, speed = 0, 0, 0
                    track_histories[obj['track_id']]['centroids'].append(centroid)
                    track_histories[obj['track_id']]['timestamps'].append(frame['timestamp'])
                    track_histories[obj['track_id']]['depths'].append(obj['depth']['mean'] if 'depth' in obj and isinstance(obj['depth'], dict) and 'mean' in obj['depth'] else 0)
                    track_histories[obj['track_id']]['speed_xs'].append(speed_x)
                    track_histories[obj['track_id']]['speed_ys'].append(speed_y)
                    track_histories[obj['track_id']]['speeds'].append(speed)
                    obj['speed_x'] = speed_x
                    obj['speed_y'] = speed_y
                    obj['speed'] = speed
                last_obj_info[int(track.track_id)] = {
                    'poly2d': obj['poly2d'],
                    'bbox': poly2d_to_bbox(obj['poly2d']),
                    'depth': obj.get('depth'),
                    'category': obj['category'],
                    'confidence': obj.get('confidence', 1.0),
                    'attributes': obj.get('attributes', {}),
                    'speed_x': obj.get('speed_x', 0),
                    'speed_y': obj.get('speed_y', 0),
                    'speed': obj.get('speed', 0)
                }
        # Draw bboxes and track_ids on frame
        img_path = os.path.join(input_dir, frame_files[frame_idx])
        img = cv2.imread(img_path)
        for obj in objects:
            if 'track_id' in obj:
                bbox = poly2d_to_bbox(obj['poly2d'])
                color = (0, 255, 0)
                img = draw_bbox_and_id(img, bbox, obj['track_id'], obj['category'], color)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), img)
        # Aggregate for output JSON
        aggregated_frames.append({
            'timestamp': frame['timestamp'],
            'objects': objects
        })
    # --- Prediction frame ---
    pred_objects = []
    for track_id, info in last_obj_info.items():
        # Use last known speed_x and speed_y to shift all poly2d points
        speed_x = track_histories[track_id]['speed_xs'][-1]
        speed_y = track_histories[track_id]['speed_ys'][-1]
        # Predict new poly2d by shifting all points by speed_x, speed_y (predicting 1 frame ahead)
        new_poly2d = [[float(p[0]) + speed_x, float(p[1]) + speed_y, p[2]] for p in info['poly2d']]
        depths = track_histories[track_id]['depths']
        if len(depths) >= 2:
            depth_diff = depths[-1] - depths[-2]
            new_depth = depths[-1] + depth_diff
        else:
            new_depth = depths[-1]
        pred_obj = {
            'track_id': track_id,
            'category': info['category'],
            'attributes': info['attributes'],
            'confidence': info['confidence'],
            'poly2d': new_poly2d,
            'depth': new_depth,
            'speed_x': speed_x,
            'speed_y': speed_y,
            'speed': track_histories[track_id]['speeds'][-1]
        }
        pred_objects.append(pred_obj)

    # Save last frame (before prediction) as image
    last_img = cv2.imread(os.path.join(input_dir, frame_files[-1]))
    # Find objects in last frame
    last_frame_objects = aggregated_frames[-1]['objects'] if aggregated_frames and 'objects' in aggregated_frames[-1] else []
    for obj in last_frame_objects:
        if 'track_id' in obj:
            bbox = poly2d_to_bbox(obj['poly2d'])
            color = (0, 255, 0)
            last_img = draw_bbox_and_id(last_img, bbox, obj['track_id'], obj['category'], color)
    last_img_path = os.path.join(output_dir, "last_frame.jpg")
    cv2.imwrite(last_img_path, last_img)
    # Save last frame JSON
    last_json_path = os.path.join(output_dir, "last_frame.json")
    with open(last_json_path, 'w') as f:
        json.dump({
            'timestamp': frames[-1]['timestamp'] if frames else None,
            'objects': last_frame_objects
        }, f, indent=4)

    # Predict 50 frames ahead, update poly2d, and draw on blank images
    num_pred_frames = 50
    # Get image shape from last frame
    last_img = cv2.imread(os.path.join(input_dir, frame_files[-1]))
    img_shape = last_img.shape if last_img is not None else (720, 1280, 3)  # fallback shape
    H, W = img_shape[0], img_shape[1]
    curr_pred_objects = pred_objects
    # For depth extrapolation, maintain a per-object depth history
    obj_depth_histories = {}
    for obj in curr_pred_objects:
        tid = obj['track_id']
        d = obj.get('depth', 0)
        if isinstance(d, dict):
            d = d.get('mean', 0)
        obj_depth_histories[tid] = [d]
    for pred_idx in range(1, num_pred_frames + 1):
        next_pred_objects = []
        for obj in curr_pred_objects:
            speed_x = obj.get('speed_x', 0)
            speed_y = obj.get('speed_y', 0)
            # Shift poly2d
            new_poly2d = [[float(p[0]) + speed_x, float(p[1]) + speed_y, p[2]] for p in obj['poly2d']]
            # Check if all points are out of frame
            out_of_frame = all((p[0] < 0 or p[0] >= W or p[1] < 0 or p[1] >= H) for p in new_poly2d)
            # Depth extrapolation
            tid = obj['track_id']
            d_hist = obj_depth_histories.get(tid, [0])
            last_depth = d_hist[-1]
            if len(d_hist) >= 2:
                prev_depth = d_hist[-2]
                new_depth = last_depth + (last_depth - prev_depth)
            else:
                new_depth = last_depth
            # If out of frame, set speeds to 0
            if out_of_frame:
                speed_x = 0
                speed_y = 0
                obj_speed = 0
            else:
                obj_speed = (speed_x ** 2 + speed_y ** 2) ** 0.5
            # Only keep objects that are not completely out of frame
            if not out_of_frame:
                next_pred_obj = {
                    'track_id': obj['track_id'],
                    'category': obj['category'],
                    'attributes': obj.get('attributes', {}),
                    'confidence': obj.get('confidence', 1.0),
                    'poly2d': new_poly2d,
                    'depth': new_depth,
                    'speed_x': speed_x,
                    'speed_y': speed_y,
                    'speed': obj_speed
                }
                next_pred_objects.append(next_pred_obj)
                # Update depth history for next step
                obj_depth_histories[tid].append(new_depth)
        # Draw on blank image
        blank_img = np.zeros(img_shape, dtype=np.uint8)
        for obj in next_pred_objects:
            bbox = poly2d_to_bbox(obj['poly2d'])
            color = (0, 0, 255)
            blank_img = draw_bbox_and_id(blank_img, bbox, obj['track_id'], obj['category'], color)
        pred_img_path = os.path.join(output_dir, f"frame_pred_{pred_idx:02d}.jpg")
        cv2.imwrite(pred_img_path, blank_img)
        # Save prediction JSON for this frame
        pred_json_path = os.path.join(output_dir, f"frame_pred_{pred_idx:02d}.json")
        with open(pred_json_path, 'w') as f:
            json.dump({
                'timestamp': f'prediction_{pred_idx}',
                'objects': next_pred_objects
            }, f, indent=4)
        # Prepare for next iteration
        curr_pred_objects = next_pred_objects

    # Save aggregated output JSON
    output_json_path = os.path.join(output_dir, "deepsort_output.json")
    with open(output_json_path, "w") as f:
        json.dump({
            'name': os.path.basename(input_dir),
            'frames': aggregated_frames
        }, f, indent=4)



if __name__ == "__main__":
    if len(sys.argv) > 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        midas_json_path = sys.argv[3]
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory.")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
        process_frames(input_dir, output_dir, midas_json_path)
    else:
        print("Usage: python video_pipe_deepsort.py <input_dir> <output_dir> <midas_json_path>")
        sys.exit(1)
