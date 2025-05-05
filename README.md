# Vehicle Occlusion Detection

This project uses YOLOv8 for vehicle detection with a focus on handling occluded objects. The model is trained on a custom dataset to detect various vehicle types and traffic objects.

## Features

- Vehicle detection with occlusion handling
- Multiple class detection (cars, trucks, buses, etc.)
- GPU-accelerated training
- System resource monitoring
- Test script for model evaluation

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in YOLO format:
   - Images in `bdd_images/train`, `bdd_images/val`, and `bdd_images/test`
   - Labels in corresponding directories
   - Update `data.yaml` with your paths

## Training

Run the training script:
```bash
python train_yolo.py
```

## Testing

Test the model on random images:
```bash
python test_model.py
```

## Model Configuration

- Model: YOLOv8n
- Image size: 640x640
- Batch size: 8
- Optimizer: AdamW
- Learning rate: 0.00005
- Augmentation: Mosaic, RandAugment

## Classes

1. bike
2. bus
3. car
4. motor
5. person
6. rider
7. traffic light
8. traffic sign
9. train
10. truck 