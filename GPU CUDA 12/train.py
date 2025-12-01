import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from model import DETRBaby
from dataset_coco import COCODataset
from tqdm import tqdm

JSON = "data/coco_cow.json"
IMGDIR = "data/images"

def collate_fn(batch):
    return batch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset = COCODataset(JSON, IMGDIR, transforms=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = DETRBaby(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    torch.backends.cudnn.benchmark = True

    epochs = 80
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = torch.stack([b["image"] for b in batch]).to(device)
            out = model(images)

            logits = out["pred_logits"]
            boxes = out["pred_boxes"]

            cls_loss = logits.mean()
            box_loss = boxes.mean()
            loss = cls_loss + box_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": float(loss)})

    torch.save(model.state_dict(), "rfdet_cow_gpu.pth")
    print("Training complete -> rfdet_cow_gpu.pth")

if __name__ == "__main__":
    main()
