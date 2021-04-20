import torch
import numpy as np


def convert_bbox_to_distance(points, bboxes):
    xs, ys = torch.unbind(points, dim=-1)
    x_min, y_min, x_max, y_max = torch.unbind(bboxes, dim=-1)

    left = xs - x_min
    right = x_max - xs
    top = ys - y_min
    bottom = y_max - ys

    return torch.stack([left, top, right, bottom], dim=-1)


def convert_distance_to_bbox(points, distances):
    xs, ys = torch.unbind(points, dim=-1)
    left, top, right, bottom = torch.unbind(distances, dim=-1)

    x_min = xs - left
    y_min = ys - top
    x_max = xs + right
    y_max = ys + bottom

    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def recover_bboxes(bboxes, oh, ow, ih, iw):
    h_scale, w_scale = oh / ih, ow / iw
    bboxes[:, [0, 2]] *= w_scale
    bboxes[:, [1, 3]] *= h_scale
    return bboxes
