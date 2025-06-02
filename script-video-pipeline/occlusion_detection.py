import sys
import json
import os
from collections import defaultdict

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def poly_to_bbox(poly2d):
    xs = [pt[0] for pt in poly2d]
    ys = [pt[1] for pt in poly2d]
    return [min(xs), min(ys), max(xs), max(ys)]

def detect_occlusion(frame_objs, iou_thresh=0.5):
    n = len(frame_objs)
    occluded_flags = [False]*n
    bboxes = []
    for obj in frame_objs:
        if obj['poly2d']:
            bbox = poly_to_bbox(obj['poly2d'])
        elif obj['bbox']:
            bbox = obj['bbox']
        else:
            bbox = None
        bboxes.append(bbox)
    for i in range(n):
        for j in range(n):
            if i == j or bboxes[i] is None or bboxes[j] is None:
                continue
            if iou(bboxes[i], bboxes[j]) > iou_thresh:
                if frame_objs[i].get('confidence', 1.0) < frame_objs[j].get('confidence', 1.0):
                    occluded_flags[i] = True
                else:
                    occluded_flags[j] = True
    return occluded_flags

def main(json_path):
    with open(json_path, 'r') as f:
        d = json.load(f)
    data=d['frames']
    for frame in data:
        objs = frame['objects'] 
        occluded = detect_occlusion(objs)
        for obj, occ in zip(objs, occluded):
            obj['attributes']['occluded'] = ("true" if bool(occ) else "false")
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=2)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python occlusion_detection.py <deepsort_output.json>')
        sys.exit(1)
    main(sys.argv[1])
