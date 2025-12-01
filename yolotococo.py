import os
import json
import glob
from PIL import Image

def convert_yolo_to_coco(img_dir, label_dir, output_json, categories):
    images = []
    annotations = []
    ann_id = 1

    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_id, img_path in enumerate(img_files, 1):
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h
        })

        label_path = os.path.join(
            label_dir,
            os.path.basename(img_path).replace(".jpg", ".txt")
        )

        if not os.path.isfile(label_path):
            continue

        with open(label_path, "r") as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.split())

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    "bbox": [
                        (x - bw/2) * w,
                        (y - bh/2) * h,
                        bw * w,
                        bh * h
                    ],
                    "area": bw * w * bh * h,
                    "iscrowd": 0
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i+1, "name": name} for i, name in enumerate(categories)
        ]
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print("COCO dataset written to:", output_json)


if __name__ == "__main__":
    convert_yolo_to_coco(
        img_dir="data/cow_dataset/images",
        label_dir="data/cow_dataset/labels",
        output_json="coco_cow.json",
        categories=["cow"]
    )
