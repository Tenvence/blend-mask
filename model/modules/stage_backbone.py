import torch.nn as nn
import torchvision.models as cv_models


class StageBackbone(nn.Module):
    def __init__(self):
        super(StageBackbone, self).__init__()

        backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
        self._freeze_bn(backbone)

        self.backbone_c5 = list(backbone.children())[-1]
        self.backbone_c4 = list(backbone.children())[-2]
        self.backbone_c3 = nn.Sequential(*list(backbone.children())[:-2])

    @staticmethod
    def _freeze_bn(backbone):
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        c3 = self.backbone_c3(x)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)
        return {'c3': c3, 'c4': c4, 'c5': c5}
