# RF-DETR Clean: Minimal DETR Training Pipeline (COCO-Style, Custom Dataset)

This repository contains a lightweight DETR-style model (“DETRBaby”) and a clean
training pipeline designed for custom object detection datasets using the COCO format.
It has been optimized for two environments:

- **Low-resource CPU systems (8 GB RAM)**  
- **GPU systems (RTX 4060 / 12GB VRAM or higher)**

This project trains on your custom cow dataset (or any dataset converted to
COCO-format JSON).

---

## 1. Project Structure

```bash
rfdetr_clean/
│
├── train.py
├── model.py
├── dataset_coco.py
├── make_coco.py
│
└── data/
├── images/ # JPG/PNG images
└── coco_cow.json # COCO-format annotation file
```

---

## 2. Requirements

### CPU-Only (Laptop)
Works with:
- Python 3.10+
- PyTorch (CPU build)
- torchvision
- pillow, tqdm, json libraries

### GPU Training (PC with RTX 4060 12GB)
Recommended versions:
- PyTorch ≥ 2.1 (CUDA 11.8 or 12.1)
- torchvision ≥ 0.16
- CUDA drivers up to date

---

## 3. Install Dependencies

### A. CPU Environment (Laptop)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow tqdm
```

# GPU Environment (RTX 4060 PC)

Choose either CUDA 11.8 or CUDA 12.1.
CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

# Prepare Your Dataset
Place your dataset here:
```bash
rfdetr_clean/data/images/
```
Then convert YOLO → COCO using your converted script:
```bash
python3 make_coco.py
```
You should see:
```bash
done: data/coco_cow.json
images=XXXX, annotations=YYYY
```

# Verify Dataset Integrity
Run this quick check:
```bash
python3 - << 'EOF'
from dataset_coco import COCODataset
ds = COCODataset("data/coco_cow.json", "data/images")
print("Dataset length =", len(ds))
EOF
```
Output should be the image count:
```bash
Dataset length = 4131
```

---

# Training on CPU
```bash
python3 train.py
```
This uses:
* Low-memory DETRBaby model
* 416×416 images
* Batch size 1
* Tiny transformer (2 encoder + 2 decoder layers)
A trained model is saved as:
```bash
rfdet_cow_cpu8gb.pth
```

# Training on GPU
Simply increase training parameters in train.py:

Recommended changes:
```bash
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

model = DETRBaby(num_classes=1).to("cuda")
```
Then run:
```bash
python3 train.py
```
Training will be 10–20× faster with much better convergence.

# Export Trained Model
Once trained:
```bash
rfdet_cow_gpu.pth
```
To export to ONNX:
```bash
python3 export_onnx.py
```

# Inference Example
A simple inference demo:
```bash
import torch
from PIL import Image
from torchvision import transforms
from model import DETRBaby

model = DETRBaby(num_classes=1)
model.load_state_dict(torch.load("rfdet_cow_gpu.pth", map_location="cpu"))
model.eval()

img = Image.open("test.jpg")
tf = transforms.Compose([
    transforms.Resize((416,416)),
    transforms.ToTensor()
])
x = tf(img).unsqueeze(0)

out = model(x)
print(out)
```

# Notes and Limitations
* DETRBaby is not a full DETR model but a reduced version for low memory.
* For best accuracy, train on GPU with larger images and larger batch sizes.
* COCO format must contain:
```
    * images
    * annotations
    * categories
```

# Recommended GPU Training Settings
RTX 4060 (12GB):
```bash
image size: 640×640
batch size: 8
epochs: 80–120
lr: 1e-4 (AdamW)
```
For maximum performance, enable:
```bash
torch.backends.cudnn.benchmark = True
```

