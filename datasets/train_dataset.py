import os

import cv2
import albumentations as alb
import torch
import torchvision.datasets as cv_datasets
import torchvision.ops as cv_ops
import pycocotools.mask as coco_mask
import numpy as np

import datasets.tools as tools
import utils.bbox_ops as bbox_ops


class TrainDataset(cv_datasets.CocoDetection):
    def __init__(self, root, year, input_size, pooler_size):
        super(TrainDataset, self).__init__(root=os.path.join(root, f'train{year}'), annFile=os.path.join(root, 'annotations', f'instances_train{year}.json'))

        self.h, self.w = input_size
        self.pooler_size = pooler_size

        self.cat_idx_list, self.cat_to_label_map, _ = tools.get_cat_label_map(self.coco, tools.COCO_CLASSES)

        self.img_transform = alb.Compose([
            alb.RandomSizedBBoxSafeCrop(width=self.w, height=self.h),
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0., p=0.8),
        ], bbox_params=alb.BboxParams(format='coco', label_fields=['class_labels']))

        self.points, self.regress_ranges = tools.encode_points_and_regress_ranges(self.h, self.w)

    def __getitem__(self, index):
        img, target = tools.load_img_target(self, index)
        img_info = self.coco.loadImgs(self.ids[index])[0]
        iw, ih = img_info['width'], img_info['height']

        class_labels, bbox_labels, mask_labels = [], [], []
        for obj in target:
            if not tools.is_correct_instance(obj, self.cat_idx_list, iw, ih):
                continue

            class_labels.append(self.cat_to_label_map[obj['category_id']])
            bbox_labels.append(obj['bbox'])

            # rle = coco_mask.frPyObjects(obj['segmentation'], ih, iw)
            # if obj['iscrowd'] == 0:
            #     rle = coco_mask.merge(rle)
            # mask = coco_mask.decode(rle)
            # mask_labels.append(mask)

        transformed = self.img_transform(image=img, bboxes=bbox_labels, class_labels=class_labels)
        # transformed = self.img_transform(image=img, masks=mask_labels, bboxes=bbox_labels, class_labels=class_labels)
        img = tools.TENSOR_TRANSFORM(transformed['image'])
        # mask_labels = transformed['masks']
        class_labels = transformed['class_labels']
        bbox_labels = transformed['bboxes']

        if len(bbox_labels) == 0:
            # For any instance with classification label 0 (background), only classification loss will be computed, without mask loss, centerness loss and bbox loss.
            # When there is no instances in an image, it doesn't matter the value of the added bbox.
            mask_labels = [np.zeros((self.h, self.w))]
            bbox_labels = [[0., 0., 10., 10.]]
            class_labels = [0]

        class_labels = torch.as_tensor(class_labels)

        # instance_mask_labels = self._generate_instance_mask_labels(mask_labels, bbox_labels)
        # instance_mask_labels = torch.as_tensor(np.array(instance_mask_labels)).float()

        bbox_labels = cv_ops.box_convert(torch.as_tensor(bbox_labels, dtype=torch.float32), in_fmt='xywh', out_fmt='xyxy')
        bbox_labels = cv_ops.clip_boxes_to_image(bbox_labels, (ih, iw))

        class_targets, distance_targets = self._encode_targets(class_labels, bbox_labels, None)
        centerness_targets = tools.encode_centerness_targets(distance_targets)

        return img, self.points, {'class': class_targets, 'distance': distance_targets, 'centerness': centerness_targets}

    def _generate_instance_mask_labels(self, mask_labels, bbox_labels):
        instance_mask_labels = []
        for mask_label, bbox_label in zip(mask_labels, bbox_labels):
            bx, by, bw, bh = bbox_label
            bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)

            roi_mask = mask_label[by:by + bh, bx:bx + bw]
            roi_mask = cv2.resize(roi_mask, dsize=(self.pooler_size, self.pooler_size))
            instance_mask_labels.append(roi_mask)
        return instance_mask_labels

    def _encode_targets(self, cls_labels, bbox_labels, instance_mask_labels):
        points = self.points.clone()
        regress_ranges = self.regress_ranges.clone()

        num_points = points.size(0)
        num_gts = cls_labels.size(0)

        regress_ranges = regress_ranges[:, None, :].repeat(1, num_gts, 1)  # [num_points, num_gts, 2]
        bbox_areas = cv_ops.box_area(bbox_labels)[None].repeat(num_points, 1)  # [num_points, num_gts]

        expanded_points = points[:, None, :].repeat(1, num_gts, 1)
        expanded_bboxes = bbox_labels[None, :, :].repeat(num_points, 1, 1)
        distance_targets = bbox_ops.convert_bbox_to_distance(expanded_points, expanded_bboxes)  # [num_points, num_gts, 4]
        # instance_mask_labels = instance_mask_labels[None, :, :, :].repeat(num_points, 1, 1, 1)  # [num_points, num_gts, roi_size, roi_size]

        # Condition 1: inside a gt bbox
        inside_gt_bbox_mask = distance_targets.min(dim=-1)[0] > 0  # [num_points, num_gts]

        # Condition 2: limit the regression range for each location
        max_regress_distance = distance_targets.max(dim=-1)[0]  # [num_points, num_gts]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])  # [num_points, num_gts]

        # If there are still more than one instances for a location, we choose the one with minimal area
        bbox_areas[inside_gt_bbox_mask == 0] = tools.INF
        bbox_areas[inside_regress_range == 0] = tools.INF
        min_area, min_area_idx = bbox_areas.min(dim=1)  # [num_points], Assign a gt to each location

        class_targets = cls_labels[min_area_idx]
        class_targets[min_area == tools.INF] = 0

        distance_targets = distance_targets[range(num_points), min_area_idx, :]
        # instance_mask_labels = instance_mask_labels[range(num_points), min_area_idx, :, :]

        return class_targets, distance_targets  # , instance_mask_labels
