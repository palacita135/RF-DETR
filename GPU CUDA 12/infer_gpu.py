import torch
from PIL import Image
from torchvision import transforms
from model import DETRBaby

IMG = "test.jpg"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on: {device}")

    model = DETRBaby(num_classes=1).to(device)
    model.load_state_dict(torch.load("rfdet_cow_gpu.pth", map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    img = Image.open(IMG).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    logits = out["pred_logits"][0]
    boxes = out["pred_boxes"][0]

    print("logits:", logits[:5])
    print("boxes:", boxes[:5])

if __name__ == "__main__":
    main()
