import torch.nn as nn
import torch.nn.functional as func
import torchvision.ops as cv_ops


class Blender(nn.Module):
    def __init__(self, pooler_size):
        super(Blender, self).__init__()

        self.pooler_size = pooler_size
        self.pooler = cv_ops.RoIAlign(output_size=pooler_size, spatial_scale=0.25, sampling_ratio=1)

    def forward(self, bases, box_proposals, attention_proposals):
        batch_size, num_proposals, num_bases, _, _ = attention_proposals.shape
        rois = self.pooler(bases, box_proposals.unbind(dim=0)).reshape(batch_size, num_proposals, num_bases, self.pooler_size, self.pooler_size)
        attention_proposals = func.interpolate(attention_proposals.flatten(1, 2), size=(self.pooler_size, self.pooler_size), mode='bilinear', align_corners=True)
        attention_proposals = attention_proposals.unflatten(1, (num_proposals, num_bases)).softmax(dim=2)
        blended_mask = (attention_proposals * rois).sum(dim=2)
        return blended_mask
