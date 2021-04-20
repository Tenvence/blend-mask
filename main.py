import argparse
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data as data

import engine
import utils.lr_lambda
from datasets import COCO_CLASSES, TrainDataset, ValDataset
from model import BlendMask, Criterion
from utils.distributed_logger import DistributedLogger


def get_args_parser():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--dataset_year', default=2014, type=int)
    parser.add_argument('--output_base_path', default='./output', type=str)
    parser.add_argument('--use_tensorboard', default=False, type=bool)
    parser.add_argument('--name', type=str)
    parser.add_argument('--random_seed', default=970423, type=int)

    parser.add_argument('--input_size_h', default=512, type=int)
    parser.add_argument('--input_size_w', default=512, type=int)
    parser.add_argument('--fpn_channels', default=256, type=int)
    parser.add_argument('--bases_module_channels', default=128, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--atten_size', default=2, type=int)
    parser.add_argument('--pooler_size', default=28, type=int)

    parser.add_argument('--focal_alpha', default=.25, type=float)
    parser.add_argument('--focal_gamma', default=2., type=float)

    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--bias_lr_mul', default=2., type=float)
    parser.add_argument('--bias_weight_decay_mul', default=0., type=float)
    parser.add_argument('--warm_up_epoch', default=1, type=int)
    parser.add_argument('--warm_up_ratio', default=1 / 3, type=float)
    parser.add_argument('--milestones', default=[7, 11], type=list)
    parser.add_argument('--step_gamma', default=0.1, type=float)

    parser.add_argument('--nms_pre', default=1000, type=int)
    parser.add_argument('--nms_cls_score_thr', default=0.05, type=int)
    parser.add_argument('--nms_iou_thr', default=0.5, type=int)

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--master_rank', default=0, type=int)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def __main__():
    args = get_args_parser()
    dist.init_process_group(backend='nccl')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.random_seed + dist.get_rank())
    torch.cuda.set_device(torch.device('cuda:{}'.format(dist.get_rank())))
    dist_logger = DistributedLogger(args.name, args.output_base_path, args.master_rank, args.use_tensorboard)

    train_dataset = TrainDataset(args.dataset_root, args.dataset_year, (args.input_size_h, args.input_size_w), args.pooler_size)
    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)

    val_dataset = ValDataset(args.dataset_root, args.dataset_year, (args.input_size_h, args.input_size_w))
    val_sampler = data.distributed.DistributedSampler(val_dataset)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

    model = BlendMask(len(COCO_CLASSES), args.fpn_channels, args.bases_module_channels, args.num_bases, args.atten_size, args.pooler_size).cuda()
    # model.load_state_dict(torch.load(f'./output/{args.name}/model/param.pth'))
    model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    criterion = Criterion(args.focal_alpha, args.focal_gamma)

    optim_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if not n.endswith('bias') and p.requires_grad]},
        {
            'params': [p for n, p in model.module.named_parameters() if n.endswith('bias') and p.requires_grad],
            'lr': args.lr * args.bias_lr_mul,
            'weight_decay': args.weight_decay * args.bias_weight_decay_mul
        }
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = utils.lr_lambda.get_warm_up_multi_step_lr_lambda(len(train_dataloader), args.warm_up_epoch, args.warm_up_ratio, args.milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    nms_cfg = {'nms_pre': args.nms_pre, 'cls_score_thr': args.nms_cls_score_thr, 'iou_thr': args.nms_iou_thr}

    for epoch_idx in range(args.epochs):
        train_sampler.set_epoch(epoch_idx)
        val_sampler.set_epoch(epoch_idx)

        engine.train_one_epoch(model, criterion, optimizer, lr_scheduler, train_dataloader, epoch_idx, dist_logger)
        # engine.val_one_epoch(model, val_dataloader, val_dataset.coco, dist_logger, epoch_idx, nms_cfg)
        # exit(-1)


if __name__ == '__main__':
    __main__()
