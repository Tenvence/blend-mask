import os

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as func
import torch.nn.utils
import torchvision.ops as cv_ops
import tqdm

import datasets.tools as tools
import utils.bbox_ops as bbox_ops


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, epoch_idx, dist_logger):
    model.train()
    scaler = amp.GradScaler()
    processor = tqdm.tqdm(data_loader, disable=not dist_logger.is_master_rank)

    epoch_losses = []
    epoch_losses_class = []
    epoch_losses_bbox = []
    epoch_losses_centerness = []
    epoch_losses_instance_mask = []

    for img, points, targets in processor:
        img = img.cuda(non_blocking=True)
        points = points.cuda(non_blocking=True)
        targets = {key: val.cuda(non_blocking=True) for key, val in targets.items()}

        with amp.autocast():
            pred = model(img, points, targets['distance'])
            losses = criterion(points, pred, targets)

            loss_class = losses['class']
            loss_bbox = losses['bbox']
            loss_centerness = losses['centerness']
            # loss_instance_mask = losses['instance_mask']

            loss = loss_class + loss_bbox + loss_centerness  # + loss_instance_mask

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        epoch_losses.append(loss.clone().detach())
        epoch_losses_class.append(loss_class.clone().detach())
        epoch_losses_bbox.append(loss_bbox.clone().detach())
        epoch_losses_centerness.append(loss_centerness.clone().detach())
        # epoch_losses_instance_mask.append(loss_instance_mask.clone().detach())

        cur_loss = dist_logger.reduce_loss(loss)
        avg_loss = dist_logger.reduce_epoch_loss(epoch_losses)
        processor.set_description(f'Epoch: {epoch_idx + 1}, cur_loss: {cur_loss:.3f}, avg_loss: {avg_loss:.3f}')

    dist_logger.save_model(model)
    # dist_logger.update_tensorboard(super_tag='loss', tag_scaler_dict={
    #     'loss': dist_logger.reduce_epoch_loss(epoch_losses),
    #     'class': dist_logger.reduce_epoch_loss(epoch_losses_class),
    #     'bbox': dist_logger.reduce_epoch_loss(epoch_losses_bbox),
    #     'centerness': dist_logger.reduce_epoch_loss(epoch_losses_centerness),
    #     # 'instance_mask': dist_logger.reduce_epoch_loss(epoch_losses_instance_mask)
    # }, idx=epoch_idx)


@torch.no_grad()
def val_one_epoch(model, data_loader, coco_gt, dist_logger, epoch_idx, nms_cfg):
    pred_instances = []
    nms_pre, cls_score_thr, iou_thr = nms_cfg['nms_pre'], nms_cfg['cls_score_thr'], nms_cfg['iou_thr']
    _, _, label_to_cat_map = tools.get_cat_label_map(coco_gt, tools.COCO_CLASSES)
    # print(label_to_cat_map)

    model.eval()
    processor = tqdm.tqdm(data_loader, disable=not dist_logger.is_master_rank)
    for img, points, img_ids in processor:
        img = img.cuda(non_blocking=True)
        points = points.cuda(non_blocking=True)
        img_info_list = coco_gt.loadImgs(img_ids.numpy())

        pred = model(img, points)
        class_pred = pred['class'].sigmoid()  # [B, num_points, num_classes]
        centerness_pred = pred['centerness'].sigmoid()  # [B, num_points]
        bbox_pred = bbox_ops.convert_distance_to_bbox(points, pred['distance'])  # [B, num_points, 4]
        # instance_mask_pred = pred['instance_mask'].sigmoid()  # [B, num_points, pooler_size, pooler_size]

        # print(class_pred.shape, centerness_pred.shape, bbox_pred.shape, instance_mask_pred.shape)
        # exit(-1)

        cls_pred_scores, cls_pred_indexes = class_pred.max(dim=-1)  # [B, num_points]

        batch_size, _, num_classes = class_pred.shape
        _, _, ih, iw = img.shape

        for batch_idx in range(batch_size):
            b_cls_pred_scores = cls_pred_scores[batch_idx]
            b_cls_pred_indexes = cls_pred_indexes[batch_idx]
            b_centerness_pred = centerness_pred[batch_idx]
            b_bbox_pred = bbox_pred[batch_idx, :]  # [num_points, 4]

            _, top_idx = (b_cls_pred_scores * b_centerness_pred).topk(nms_pre)
            top_class_pred_scores = b_cls_pred_scores[top_idx]
            top_class_pred_indexes = b_cls_pred_indexes[top_idx]
            top_centerness_pred = b_centerness_pred[top_idx]
            top_bbox_pred = b_bbox_pred[top_idx, :]  # [topk, 4]

            nms_scores = top_class_pred_scores * top_centerness_pred
            top_bbox_pred = cv_ops.clip_boxes_to_image(top_bbox_pred, size=(ih, iw))

            valid_mask = top_class_pred_scores > cls_score_thr
            valid_class_pred_scores = top_class_pred_scores[valid_mask]
            valid_class_pred_indexes = top_class_pred_indexes[valid_mask]
            valid_nms_scores = nms_scores[valid_mask]
            valid_bbox_pred = top_bbox_pred[valid_mask, :]

            keep_idx = cv_ops.batched_nms(valid_bbox_pred, valid_nms_scores, valid_class_pred_indexes, iou_thr)
            keep_class_pred_scores = valid_class_pred_scores[keep_idx]
            keep_class_pred_indexes = valid_class_pred_indexes[keep_idx]
            keep_bbox_pred = valid_bbox_pred[keep_idx, :]

            oh, ow = img_info_list[batch_idx]['height'], img_info_list[batch_idx]['width']
            keep_bbox_pred = bbox_ops.recover_bboxes(keep_bbox_pred, oh, ow, ih, iw)
            keep_bbox_pred = cv_ops.box_convert(keep_bbox_pred, in_fmt='xyxy', out_fmt='xywh')

            for cls_score, cls_idx, bbox in zip(keep_class_pred_scores, keep_class_pred_indexes, keep_bbox_pred):
                # poly = coco_mask.frPyObjects(poly.permute(1, 0).reshape(1, -1).detach().cpu().double().numpy(), oh, ow)
                # rle = coco_mask.merge(poly)
                # rle['counts'] = rle['counts'].decode('utf-8')

                pred_instances.append({
                    'image_id': int(img_ids[batch_idx]),
                    'category_id': label_to_cat_map[int(cls_idx) + 1],
                    'bbox': [float(str('%.1f' % coord)) for coord in bbox.tolist()],
                    # 'segmentation': rle,
                    'score': float(str('%.1f' % cls_score))
                })

    dist_logger.save_pred_instances_local_rank(pred_instances)
    dist_logger.save_val_file()
    dist_logger.evaluate(coco_gt)
