import torch
from model import DETRNano

def main():
    model = DETRNano(num_classes=1)
    model.load_state_dict(torch.load("rfdet_cow.pth", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model,
        dummy,
        "rfdet_cow.onnx",
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=17,
        dynamic_axes={"images": {0: "batch"}}
    )

    print("Exported rfdet_cow.onnx")

if __name__ == "__main__":
    main()
