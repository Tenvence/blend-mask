import torch.nn as nn
import torch.nn.functional as func
import torchvision.ops as cv_ops


class FeaturePyramidNet(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNet, self).__init__()

        # There are no Norm layers in FPN
        self.fpn = cv_ops.FeaturePyramidNetwork(in_channels_list, out_channels)
        self.conv_p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # TODO: parameters initialization for conv_p6 and conv_p7

    def forward(self, stage_backbone_features_dict):
        fpn_out = self.fpn(stage_backbone_features_dict)
        p3 = fpn_out['c3']
        p4 = fpn_out['c4']
        p5 = fpn_out['c5']
        p6 = self.conv_p6(p5)
        p7 = self.conv_p7(func.relu(p6))

        return {'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, 'p7': p7}
