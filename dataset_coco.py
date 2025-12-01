import json
import os
from PIL import Image

class COCODataset:
    def __init__(self, json_path, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(json_path, "r") as f:
            coco = json.load(f)

        # COCO keys must exist
        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]

        # group annotations by image_id
        self.ann_by_img = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.ann_by_img.setdefault(img_id, []).append(ann)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]

        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        anns = self.ann_by_img.get(img_id, [])

        return {
            "image": img,
            "annotations": anns,
            "image_id": img_id
        }
