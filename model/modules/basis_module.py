import torch
import torch.nn as nn
import torch.nn.functional as func


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class AuxSemanticSegHead(nn.Module):
    def __init__(self, fpn_channels, inner_channels, num_classes):
        super(AuxSemanticSegHead, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(fpn_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, num_classes + 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class BasisModule(nn.Module):
    def __init__(self, in_channels_list, inner_channels, num_basis):
        super(BasisModule, self).__init__()

        self.refine_block_list = nn.ModuleList([ConvBlock(in_channel, inner_channels) for in_channel in in_channels_list])

        self.tower_conv_seq = nn.Sequential(*list([ConvBlock(inner_channels, inner_channels) for _ in range(3)]))
        self.tower_block = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(inner_channels, inner_channels),
            nn.Conv2d(inner_channels, num_basis, kernel_size=1)
        )

    def forward(self, in_features):
        refined_features_list = [refine_block(in_feature) for refine_block, in_feature in zip(self.refine_block_list, in_features)]

        bottom_shape = refined_features_list[0].shape[2:]
        refined_features_list = [func.interpolate(refined_feature, size=bottom_shape, mode='bilinear', align_corners=False) for refined_feature in refined_features_list]

        entire_refined_feature = torch.stack(refined_features_list, dim=0).sum(dim=0)

        bases = self.tower_conv_seq(entire_refined_feature)
        bases = self.tower_block(bases)

        return bases
