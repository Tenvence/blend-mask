import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.ops as cv_ops

import utils.bbox_ops as bbox_ops


class Criterion(nn.Module):
    def __init__(self, focal_alpha, focal_gamma):
        super(Criterion, self).__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(self, points, pred, targets):
        class_pred = pred['class']
        centerness_pred = pred['centerness']
        distance_pred = pred['distance']
        # instance_mask_pred = pred['instance_mask']

        class_targets = targets['class']
        centerness_targets = targets['centerness']
        distance_targets = targets['distance']
        # instance_mask_targets = targets['instance_mask']

        _, _, num_classes = class_pred.shape
        # _, _, pooler_size, _ = instance_mask_targets.shape

        positive_idx = torch.nonzero(class_targets.reshape(-1)).reshape(-1)
        pos_points = points.reshape(-1, 2)[positive_idx]

        pos_distance_pred = distance_pred.reshape(-1, 4)[positive_idx]  # [num_positives, 4]
        pos_distance_targets = distance_targets.reshape(-1, 4)[positive_idx]  # [num_positives, 4]

        pos_centerness_pred = centerness_pred.reshape(-1)[positive_idx]  # [num_positives]
        pos_centerness_targets = centerness_targets.reshape(-1)[positive_idx]  # [num_positives]

        # pos_instance_mask_pred = instance_mask_pred.reshape(-1, pooler_size, pooler_size)[positive_idx]  # [num_positives, pooler_size, pooler_size]
        # pos_instance_mask_targets = instance_mask_targets.reshape(-1, pooler_size, pooler_size)[positive_idx]  # [num_positives, pooler_size, pooler_size]

        loss_bbox = self._compute_bbox_loss(pos_points, pos_distance_pred, pos_distance_targets, pos_centerness_targets)
        loss_centerness = func.binary_cross_entropy_with_logits(pos_centerness_pred, pos_centerness_targets)

        bg_targets, fg_class_targets = self._get_onehot_bg_fg_targets(num_classes, class_targets)
        loss_cls = self._compute_cls_loss(class_pred, bg_targets, fg_class_targets)

        # loss_instance_mask = self._compute_instance_mask_loss(pos_instance_mask_pred, pos_instance_mask_targets)

        return {'class': loss_cls, 'bbox': loss_bbox, 'centerness': loss_centerness}  # , 'instance_mask': loss_instance_mask}

    @staticmethod
    def _get_onehot_bg_fg_targets(num_classes, class_targets):
        class_targets = func.one_hot(class_targets, num_classes=num_classes + 1).float()
        bg_targets = class_targets[..., 0]
        fg_class_targets = class_targets[..., 1:]
        return bg_targets, fg_class_targets

    def _compute_cls_loss(self, class_pred, bg_targets, fg_class_targets):
        norm = (1. - bg_targets).sum()
        loss_cls = cv_ops.sigmoid_focal_loss(class_pred, fg_class_targets, self.alpha, self.gamma, reduction='sum') / norm
        return loss_cls

    @staticmethod
    def _compute_bbox_loss(points, distance_pred, distance_targets, centerness_targets):
        decoded_bbox_pred = bbox_ops.convert_distance_to_bbox(points, distance_pred)
        decoded_bbox_targets = bbox_ops.convert_distance_to_bbox(points, distance_targets)

        iou_loss = -cv_ops.box_iou(decoded_bbox_pred, decoded_bbox_targets).diagonal().clamp(min=1e-6).log()
        loss_bbox = (centerness_targets * iou_loss).sum() / centerness_targets.sum()

        return loss_bbox

    @staticmethod
    def _compute_semantic_mask_loss(semantic_mask_pred, semantic_mask_targets):
        output_semantic_mask_size = semantic_mask_pred.shape[2:]
        semantic_mask_targets = semantic_mask_targets.unsqueeze(dim=1)
        semantic_mask_targets = func.interpolate(semantic_mask_targets, output_semantic_mask_size)
        semantic_mask_targets = semantic_mask_targets.squeeze()

        loss_semantic_mask = func.cross_entropy(semantic_mask_pred, semantic_mask_targets.long())

        return loss_semantic_mask

    @staticmethod
    def _compute_instance_mask_loss(instance_mask_pred, instance_mask_targets):
        loss_instance_mask = func.binary_cross_entropy_with_logits(instance_mask_pred, instance_mask_targets.float())
        return loss_instance_mask
