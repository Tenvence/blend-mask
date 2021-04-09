import copy

import numpy as np
import torch.nn as nn


class FcosHead(nn.Module):
    def __init__(self, num_channels, num_classes, attention_len):
        super(FcosHead, self).__init__()

        conv_seq_block = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU())
        conv_seq = nn.Sequential(*list([copy.deepcopy(conv_seq_block) for _ in range(4)]))

        self.cla_conv_seq = copy.deepcopy(conv_seq)
        self.class_conv = nn.Conv2d(num_channels, num_classes, kernel_size=3, padding=1)
        self.centerness_conv = nn.Conv2d(num_channels, out_channels=1, kernel_size=3, padding=1)

        self.reg_conv_seq = copy.deepcopy(conv_seq)
        self.distances_reg_conv = nn.Conv2d(num_channels, out_channels=4, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(num_channels, out_channels=attention_len, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        class_bias_prior = 0.01
        class_bias_init = float(-np.log((1 - class_bias_prior) / class_bias_prior))

        for m in self.cla_conv_seq:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        nn.init.normal_(self.class_conv.weight, std=0.01)
        nn.init.constant_(self.class_conv.bias, val=class_bias_init)

        nn.init.normal_(self.centerness_conv.weight, std=0.01)
        nn.init.constant_(self.centerness_conv.bias, val=0)

        for m in self.reg_conv_seq:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        nn.init.normal_(self.distances_reg_conv.weight, std=0.01)
        nn.init.constant_(self.distances_reg_conv.bias, val=0)

        nn.init.normal_(self.attention_conv.weight, std=0.01)
        nn.init.constant_(self.attention_conv.bias, val=0)

    def forward(self, x):
        cla_feature = self.cla_conv_seq(x)
        class_pred = self.class_conv(cla_feature)
        centerness_pred = self.centerness_conv(cla_feature)

        reg_feature = self.reg_conv_seq(x)
        distances_pred = self.distances_reg_conv(reg_feature)
        attentions_pred = self.attention_conv(reg_feature)

        return {'class': class_pred, 'centerness': centerness_pred, 'distances': distances_pred, 'attentions': attentions_pred}
