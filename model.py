import torch
import torch.nn as nn
import torchvision.models as models


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # MobileNetV2 is extremely light
        net = models.mobilenet_v2(weights=None)
        self.features = net.features
        self.out_channels = 128  # we will reduce with conv

        # Reduce channels to save memory
        self.reduce = nn.Conv2d(1280, self.out_channels, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim=128, nheads=4, enc_layers=2, dec_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

    def forward(self, src, query_embed):
        memory = self.encoder(src)
        tgt = query_embed.unsqueeze(0).repeat(src.size(0), 1, 1)
        out = self.decoder(tgt, memory)
        return out


class DETRBaby(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        hidden_dim = 128

        self.backbone = TinyBackbone()

        self.input_proj = nn.Conv2d(self.backbone.out_channels, hidden_dim, kernel_size=1)

        self.transformer = SimpleTransformer(
            hidden_dim=hidden_dim,
            nheads=4,
            enc_layers=2,
            dec_layers=2
        )

        # 50 queries only (DETR uses 100; we use half)
        self.query_embed = nn.Embedding(50, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # 1 class + no-object
        self.bbox_embed = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        bs = x.size(0)

        # backbone
        fmap = self.backbone(x)
        fmap = self.input_proj(fmap)

        # flatten for transformer
        h, w = fmap.shape[-2:]
        src = fmap.flatten(2).permute(0, 2, 1)

        hs = self.transformer(src, self.query_embed.weight)

        outputs_class = self.class_embed(hs)
        outputs_coord = torch.sigmoid(self.bbox_embed(hs))

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }
