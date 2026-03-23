import timm
import torch.nn as nn
import torch.nn.functional as F

class SwinReID(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Swin Backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True
        )

        # 去掉分类头
        self.backbone.head = nn.Identity()

        # 特征维度（Swin-base = 1024）
        self.feat_dim = 1024

        # BNNeck（ReID标准）
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)

        # 分类器
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)

    def forward(self, x):
        feat = self.backbone(x)   # [B, 1024]

        feat_bn = self.bottleneck(feat)

        if self.training:
            cls_score = self.classifier(feat_bn)
            return cls_score, feat
        else:
            return F.normalize(feat, dim=1)