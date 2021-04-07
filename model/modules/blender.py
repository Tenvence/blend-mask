import torch.nn as nn


class Blender(nn.Module):
    def __init__(self):
        super(Blender, self).__init__()

    def forward(self, x):
        return x
