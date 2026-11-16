"""Two architecturally different models: SegFormer (hierarchical transformer + all-MLP decoder)
and DeepLabV3+ (CNN encoder + atrous spatial pyramid pooling decoder)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormer(nn.Module):
    def __init__(self, nc, variant="nvidia/mit-b2", pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                variant, num_labels=nc, ignore_mismatched_sizes=True)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(variant, num_labels=nc))

    def forward(self, x):
        lg = self.net(pixel_values=x).logits  # 1/4 resolution
        return F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def groups(self):
        enc = list(self.net.segformer.parameters())
        ids = {id(p) for p in enc}
        return enc, [p for p in self.parameters() if id(p) not in ids]


class DeepLab(nn.Module):
    def __init__(self, nc, encoder="resnet50", pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                                     classes=nc)

    def forward(self, x):
        return self.net(x)

    def groups(self):
        enc = list(self.net.encoder.parameters())
        ids = {id(p) for p in enc}
        return enc, [p for p in self.parameters() if id(p) not in ids]


def build_model(cfg, nc, pretrained=True):
    if cfg["model"] == "segformer":
        return SegFormer(nc, cfg.get("variant", "nvidia/mit-b2"), pretrained)
    if cfg["model"] == "deeplab":
        return DeepLab(nc, cfg.get("encoder", "resnet50"), pretrained)
    raise ValueError(cfg["model"])


def profile(model, h, w, device):
    """Parameters, trainable parameters, FLOPs of one forward pass (FLOPs = 2 x MACs)."""
    from torch.utils.flop_counter import FlopCounterMode
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return params, trainable, fc.get_total_flops()
