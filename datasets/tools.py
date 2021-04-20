import os

import cv2
import torch
import torchvision.datasets as cv_datasets
import torchvision.transforms as cv_transforms

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

INF = 1e8
STRIDES = [8, 16, 32, 64, 128]
REGRESS_RANGES = [(-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)]

TENSOR_TRANSFORM = cv_transforms.Compose([
    cv_transforms.ToTensor(),
    cv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def get_cat_label_map(coco, classes):
    cat_idx_list = coco.getCatIds(catNms=classes)
    cat_to_label_map = {cat_idx: i + 1 for i, cat_idx in enumerate(cat_idx_list)}
    label_to_cat_map = {i + 1: cat_idx for i, cat_idx in enumerate(cat_idx_list)}
    return cat_idx_list, cat_to_label_map, label_to_cat_map


def load_img_target(dataset: cv_datasets.CocoDetection, index: int):
    coco = dataset.coco
    img_id = dataset.ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)

    target = coco.loadAnns(ann_ids)
    img = cv2.imread(os.path.join(dataset.root, coco.loadImgs(img_id)[0]['file_name']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, target


def get_level_points(h, w, stride):
    fm_h, fm_w = h // stride, w // stride
    x_range = torch.arange(0, fm_w * stride, stride)
    y_range = torch.arange(0, fm_h * stride, stride)

    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1) + stride // 2

    return points


def encode_points_and_regress_ranges(h, w):
    points, regress_ranges = [], []
    for stride, regress_range in zip(STRIDES, REGRESS_RANGES):
        level_points = get_level_points(h, w, stride)
        points.append(level_points)

        regress_range = torch.tensor(regress_range)[None].repeat(level_points.size(0), 1)
        regress_ranges.append(regress_range)

    points = torch.cat(points, dim=0)  # [num_points, 2]
    regress_ranges = torch.cat(regress_ranges, dim=0)  # [num_points, 2]

    return points, regress_ranges


def encode_centerness_targets(distance_targets):
    left_right = distance_targets[:, [0, 2]]
    left_right_min = left_right.min(dim=-1)[0]
    left_right_max = left_right.max(dim=-1)[0]

    top_bottom = distance_targets[:, [1, 3]]
    top_bottom_min = top_bottom.min(dim=-1)[0]
    top_bottom_max = top_bottom.max(dim=-1)[0]

    centerness_targets = left_right_min / left_right_max * top_bottom_min / top_bottom_max

    return torch.sqrt(centerness_targets)


def is_correct_instance(obj, cat_idx_list, iw, ih):
    if obj.get('ignore', False):
        return False
    x1, y1, bw, bh = obj['bbox']
    inter_w = max(0, min(x1 + bw, iw) - max(x1, 0))
    inter_h = max(0, min(y1 + bh, ih) - max(y1, 0))
    if inter_w * inter_h == 0:
        return False
    if obj['area'] <= 0 or bw < 8 or bh < 8:
        return False
    if obj['category_id'] not in cat_idx_list:
        return False
    return True


def generate_semantic_mask_labels(class_labels, mask_labels):
    semantic_mask_labels = torch.zeros_like(mask_labels[0])
    for class_label, mask_label in zip(class_labels, mask_labels):
        class_mask = class_label * mask_label
        semantic_mask_labels = torch.where(class_mask == 0, semantic_mask_labels, class_mask)
    return semantic_mask_labels.float()
