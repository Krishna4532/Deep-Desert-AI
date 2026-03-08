"""
src/models/deeplabv3plus.py
────────────────────────────
DeepLabV3+ with a ResNet-50 backbone.

Architecture:
    Input
      └─ ResNet-50 Backbone
           ├─ low_level_features   [B, 256,  H/4,  W/4]
           └─ high_level_features  [B, 2048, H/32, W/32]
                  └─ ASPP  →  [B, 256, H/32, W/32]
                       └─ Decoder (fuse + upsample)  →  [B, C, H, W]

Note: Standard ResNet-50 strides (no dilation patching) for
compatibility with all input sizes including small CPU test images.
All spatial reconciliation is done with F.interpolate in the decoder.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      stride=stride,
                      padding=padding * dilation,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(self.gap(x), size=size,
                             mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256,
                 dilations=(6, 12, 18), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0),
            *[ASPPConv(in_ch, out_ch, d) for d in dilations],
            ASPPPooling(in_ch, out_ch),
        ])
        n = len(dilations) + 2
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class ResNet50Backbone(nn.Module):
    """
    Standard ResNet-50 (no dilation patching).
    Exposes:
        low_level  — after layer1  [B, 256,  H/4,  W/4]
        high_level — after layer4  [B, 2048, H/32, W/32]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x          = self.stem(x)
        low_level  = self.layer1(x)
        x          = self.layer2(low_level)
        x          = self.layer3(x)
        high_level = self.layer4(x)
        return {"low_level": low_level, "high_level": high_level}


class Decoder(nn.Module):
    def __init__(self, low_ch=256, aspp_ch=256,
                 out_ch=256, num_classes=9, dropout=0.1):
        super().__init__()
        self.low_proj = ConvBNReLU(low_ch, 48, kernel_size=1, padding=0)
        self.refine   = nn.Sequential(
            ConvBNReLU(aspp_ch + 48, out_ch),
            nn.Dropout(dropout),
            ConvBNReLU(out_ch, out_ch),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Conv2d(out_ch, num_classes, 1)

    def forward(self, low_level: torch.Tensor,
                aspp_out: torch.Tensor) -> torch.Tensor:
        low     = self.low_proj(low_level)
        aspp_up = F.interpolate(aspp_out, size=low.shape[-2:],
                                mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        return self.classifier(self.refine(x))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ — returns logits at input resolution."""

    def __init__(self, num_classes=9, pretrained_backbone=True,
                 output_stride=16, aspp_dilations=(6, 12, 18), dropout=0.1):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.aspp     = ASPP(in_ch=2048, out_ch=256,
                             dilations=aspp_dilations, dropout=dropout)
        self.decoder  = Decoder(low_ch=256, aspp_ch=256,
                                out_ch=256, num_classes=num_classes,
                                dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats      = self.backbone(x)
        aspp_out   = self.aspp(feats["high_level"])
        logits     = self.decoder(feats["low_level"], aspp_out)
        return F.interpolate(logits, size=input_size,
                             mode="bilinear", align_corners=False)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def parameter_groups(self, base_lr: float, backbone_multiplier: float = 0.1):
        return [
            {"params": list(self.backbone.parameters()),
             "lr": base_lr * backbone_multiplier},
            {"params": list(self.aspp.parameters()) +
                       list(self.decoder.parameters()),
             "lr": base_lr},
        ]


def build_model(cfg: dict) -> DeepLabV3Plus:
    m = cfg["model"]
    return DeepLabV3Plus(
        num_classes         = cfg["data"]["num_classes"],
        pretrained_backbone = m["pretrained_backbone"],
        output_stride       = m["output_stride"],
        aspp_dilations      = m["aspp_dilations"],
        dropout             = m["dropout"],
    )


if __name__ == "__main__":
    for h, w in [(224, 224), (512, 512)]:
        model = DeepLabV3Plus(num_classes=9, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 9, h, w), f"Shape mismatch: {out.shape}"
        print(f"  [{h}x{w}] output={out.shape}  OK")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
