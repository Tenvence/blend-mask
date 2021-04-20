import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.tensorboard as tensorboard
import pycocotools.cocoeval as coco_eval


class DistributedLogger:
    def __init__(self, name, output_base_path, master_rank, use_tensorboard):
        self.name = name
        self.output_base_path = output_base_path

        self.master_rank = master_rank
        self.is_master_rank = dist.get_rank() == self.master_rank

        self.use_tensorboard = self.is_master_rank and use_tensorboard
        self.tensorboard_logger = tensorboard.SummaryWriter(comment=name) if self.use_tensorboard else None

        self.model_path, self.val_path = self._build_file_structure()

    def _build_file_structure(self):
        if not os.path.exists(self.output_base_path) and self.is_master_rank:
            os.mkdir(self.output_base_path)

        output_path = os.path.join(self.output_base_path, self.name)
        if not os.path.exists(output_path) and self.is_master_rank:
            os.mkdir(output_path)

        model_path = os.path.join(output_path, 'model')
        if not os.path.exists(model_path) and self.is_master_rank:
            os.mkdir(model_path)

        val_path = os.path.join(output_path, 'val')
        if not os.path.exists(val_path) and self.is_master_rank:
            os.mkdir(val_path)

        return model_path, val_path

    def save_model(self, model):
        if not self.is_master_rank:
            return

        torch.save(model.module, os.path.join(self.model_path, 'model.pkl'))
        torch.save(model.module.state_dict(), os.path.join(self.model_path, 'param.pth'))

    def reduce_loss(self, loss):
        dist.reduce(loss, dst=self.master_rank, op=dist.reduce_op.SUM)
        return loss.item() / dist.get_world_size()

    def reduce_epoch_loss(self, loss_list):
        avg_loss_one_device = torch.stack(loss_list, dim=-1).mean()
        return self.reduce_loss(avg_loss_one_device)

    def update_tensorboard(self, super_tag, tag_scaler_dict, idx):
        if not self.use_tensorboard or not self.is_master_rank:
            return

        for tag, scaler in tag_scaler_dict.items():
            self.tensorboard_logger.add_scalar(super_tag + '/' + tag, scaler, idx)

    def save_pred_instances_local_rank(self, pred_instances):
        local_rank = dist.get_rank()
        np.save(os.path.join(self.val_path, f'tmp_pred_instances_rank_{local_rank}.npy'), pred_instances)

    def save_val_file(self):
        if not self.is_master_rank:
            return

        time.sleep(3.)  # wait all threads to finish
        pred_instances = []

        for local_rank in range(dist.get_world_size()):
            tmp_pre_instances_file = os.path.join(self.val_path, f'tmp_pred_instances_rank_{local_rank}.npy')
            pred_instances.extend(np.load(tmp_pre_instances_file, allow_pickle=True))
            os.remove(tmp_pre_instances_file)

        json.dump(pred_instances, open(os.path.join(self.val_path, 'val.json'), 'w'))

    def evaluate(self, coco_gt):
        if not self.is_master_rank:
            return

        val_file_path = os.path.join(self.val_path, 'val.json')
        if len(json.load(open(val_file_path, 'r'))) == 0:
            print('no prediction!')
        else:
            coco_evaluator = coco_eval.COCOeval(cocoGt=coco_gt, cocoDt=coco_gt.loadRes(val_file_path), iouType='bbox')
            coco_evaluator.evaluate()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        # self.update_tensorboard(super_tag='bbox-val', tag_scaler_dict={
        #     'AP': b_ap * 100, 'AP@50': b_ap50 * 100, 'AP@75': b_ap75 * 100,
        #     'AP@S': b_ap_s * 100, 'AP@M': b_ap_m * 100, 'AP@L': b_ap_l * 100
        # }, idx=epoch_idx)
        #
        # self.update_tensorboard(super_tag='segm-val', tag_scaler_dict={
        #     'AP': s_ap * 100, 'AP@50': s_ap50 * 100, 'AP@75': s_ap75 * 100,
        #     'AP@S': s_ap_s * 100, 'AP@M': s_ap_m * 100, 'AP@L': s_ap_l * 100
        # }, idx=epoch_idx)
