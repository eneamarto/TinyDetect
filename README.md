# TinyDetect

Lightweight object detection using TinyLlama + Vision Transformer.

## What it does

Detects people and cars in images with bounding boxes.

## Install

```bash
torch
transformers
peft
PIL
tqdm
```

## Usage

### Train
```python
from main import TrainingConfig, train_model

config = TrainingConfig(
    num_epochs=10,
    batch_size=16
)
train_model(config)
```

### Detect
```python
from main import DetectionInference

detector = DetectionInference("path/to/model")
result = detector.detect("image.jpg")
print(result)
# Output: "I found 2 objects: person at (120,50,200,180), car at (300,100,450,200)."
```

## Model

- **TinyLlama 1.1B** - Language model
- **ViT-Base** - Image encoder  
- **LoRA** - Efficient fine-tuning
- **COCO 2017** - Training data

## Output

Returns detected objects with coordinates:
```
"I found 3 objects: person at <x1,y1,x2,y2>, car at <x1,y1,x2,y2>."
```
