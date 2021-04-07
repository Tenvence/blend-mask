import torch.nn as nn


class BlendMask(nn.Module):
    def __init__(self):
        super(BlendMask, self).__init__()

    def forward(self, x):
        return x
