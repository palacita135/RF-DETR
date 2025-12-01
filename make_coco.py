import os
import json
from PIL import Image

IMAGEDIR = "data/images"
LABELDIR = "../data/cow_dataset/train/labels"   # adjust if wrong
OUT = "data/coco_cow.json"

images = []
annotations = []

ann_id = 1

image_files = sorted(os.listdir(IMAGEDIR))

for i, fname in enumerate(image_files):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_id = i + 1
    img_path = os.path.join(IMAGEDIR, fname)

    # load image size
    w, h = Image.open(img_path).size

    images.append({
        "id": img_id,
        "file_name": fname,
        "width": w,
        "height": h
    })

    # label file
    label_name = os.path.splitext(fname)[0] + ".txt"
    label_path = os.path.join(LABELDIR, label_name)

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.split())

            # YOLO to COCO
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            ww = bw * w
            hh = bh * h

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, ww, hh],
                "area": ww * hh,
                "iscrowd": 0
            })
            ann_id += 1

coco = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "cow"}]
}

with open(OUT, "w") as f:
    json.dump(coco, f)

print("done:", OUT)
print(f"images={len(images)}, annotations={len(annotations)}")
