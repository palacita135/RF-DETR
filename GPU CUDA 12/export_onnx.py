import torch
from model import DETRBaby
from torchvision import transforms
import numpy as np

def main():
    model = DETRBaby(num_classes=1)
    model.load_state_dict(torch.load("rfdet_cow_gpu.pth", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model,
        dummy,
        "rfdet_cow.onnx",
        opset_version=17,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "images": {0: "batch"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }
    )

    print("Exported to rfdet_cow.onnx")

if __name__ == "__main__":
    main()
