"""
Deepfake image classifier model.
Uses a pretrained EfficientNet-B0 backbone (fast to train, good accuracy,
lighter on GPU memory than Xception -- good choice for a laptop RTX GPU).

Output: single logit -> sigmoid -> probability that the image is FAKE.
"""
import torch
import torch.nn as nn
import timm


class DeepfakeImageClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, dropout=0.4):
        super().__init__()

        # Load pretrained backbone, remove its classifier head (num_classes=0)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # single logit
        )

    def forward(self, x):
        feats = self.backbone(x)          # [B, feat_dim]
        logit = self.classifier(feats)    # [B, 1]
        return logit.squeeze(1)           # [B]


def build_model(device="cuda"):
    model = DeepfakeImageClassifier()
    return model.to(device)


if __name__ == "__main__":
    # quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: [4]
    print("Model loaded successfully on:", device)
