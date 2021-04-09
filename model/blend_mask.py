import torch
import torch.nn as nn

import model.modules as modules


class BlendMask(nn.Module):
    def __init__(self, num_classes, num_basis, attention_size, num_channels):
        super(BlendMask, self).__init__()

        self.num_classes = num_classes
        self.num_basis = num_basis
        self.attention_size = attention_size
        self.attention_len = num_basis * attention_size * attention_size

        self.backbone = modules.StageBackbone()
        self.fpn = modules.FeaturePyramidNet(in_channels_list=[512, 1024, 2048], out_channels=num_channels)

        self.focs_head = modules.FcosHead(num_channels, num_classes, self.attention_len)
        self.fcos_distances_scales = [nn.Parameter(torch.tensor(1., dtype=torch.float)) for _ in range(5)]

        self.basis_module = modules.BasisModule(in_channels_list=[num_channels, num_channels, num_channels], inner_dim=num_channels, num_basis=num_basis)
        self.aux_semantic_seg_head = modules.AuxSemanticSegHead(num_channels, num_channels, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        stage_features_dict = self.backbone(x)
        fpn_features_dict = self.fpn(stage_features_dict)

        class_pred_level_list, centerness_pred_level_list, distances_pred_level_list, attentions_pred_level_list = [], [], [], []
        for idx, (distance_scale, fpn_feature) in enumerate(zip(self.fcos_distances_scales, fpn_features_dict.values())):
            fcos_head_out = self.focs_head(fpn_feature)

            class_pred = fcos_head_out['class']
            centerness_pred = fcos_head_out['centerness']
            distances_pred = torch.exp(fcos_head_out['distances'] * distance_scale)
            attentions_pred = fcos_head_out['attentions']

            class_pred_level_list.append(class_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            centerness_pred_level_list.append(centerness_pred.permute(0, 2, 3, 1).reshape(batch_size, -1))
            distances_pred_level_list.append(distances_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
            attentions_pred_level_list.append(attentions_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_basis, self.attention_size, self.attention_size))

        class_pred = torch.cat(class_pred_level_list, dim=1)
        centerness_pred = torch.cat(centerness_pred_level_list, dim=1)
        distances_pred = torch.cat(distances_pred_level_list, dim=1)
        attentions_pred = torch.cat(attentions_pred_level_list, dim=1)

        basis = self.basis_module([fpn_features_dict['p3'], fpn_features_dict['p4'], fpn_features_dict['p5']])
        semantic_seg = self.aux_semantic_seg_head(fpn_features_dict['p3'])
        print(basis.shape, semantic_seg.shape)

        return class_pred, centerness_pred, distances_pred, attentions_pred
