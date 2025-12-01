import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DETRBaby
from dataset_coco import COCODataset
import torch.optim as optim

JSON = "data/coco_cow.json"
IMGDIR = "data/images"

def collate_fn(batch):
    return batch

def main():
    device = "cpu"

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])

    dataset = COCODataset(JSON, IMGDIR, transforms=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = DETRBaby(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(40):
        for batch in loader:
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

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "rfdet_cow_cpu8gb.pth")
    print("Training finished. Saved rfdet_cow_cpu8gb.pth")

if __name__ == "__main__":
    main()
