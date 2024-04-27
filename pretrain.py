import argparse
import math
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torch.distributed as dist

import misc

import moco.builder_dist
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Distributed
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation  for self supervised training (image-based or graph-based or seq-based)')
parser.add_argument('--pre-dataset', default='SLR', type=str)

# contrast specific configs:
parser.add_argument('--contrast-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--contrast-k', default=32768, type=int,
                    help='queue size; number of negative keys (default: 16384)')
parser.add_argument('--contrast-m', default=0.999, type=float,
                    help='contrast momentum of updating key encoder (default: 0.999)')
parser.add_argument('--contrast-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--teacher-t', default=0.05, type=float,
                    help='softmax temperature (default: 0.05)')
parser.add_argument('--student-t', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--KT-weight', default=1.0, type=float,
                    help='weight of sim loss (default: 1.0)')
parser.add_argument('--inter-weight', default=0.5, type=float,
                    help='weight of inter consistency loss (default: 0.5)')
parser.add_argument('--topk', default=1024, type=int,
                    help='number of contrastive context')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--inter-dist', action='store_true',
                    help='use inter distillation loss')



def main():
    args = parser.parse_args()
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # pretraining dataset
    from options import options_pretraining as options
    if args.pre_dataset == 'SLR':
        opts = options.opts_SLR_cross_subject()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation

    # create model
    print("=> creating model")

    model = moco.builder_dist.MoCo(args.skeleton_representation, opts.num_class,
                                       args.contrast_dim, args.contrast_k, args.contrast_m, args.contrast_t,
                                       args.teacher_t, args.student_t, args.KT_weight, args.topk, args.mlp, inter_weight=args.inter_weight,
                                       inter_dist=args.inter_dist)
    print("options", opts.train_feeder_args)
    print(model)

    model.to(device)
    if args.distributed:
        model = nn.parallel.distributed.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        print('Distributed data parallel model used')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## Data loading code
    train_dataset = get_pretraining_set(opts)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers,
        worker_init_fn=worker_init_fn, pin_memory=True, sampler=train_sampler, drop_last=True)

    if global_rank == 0:
        writer = SummaryWriter(log_dir=args.checkpoint_path)
    else:
        writer = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss_joint, loss_motion, loss_sim, loss_inter, top1_joint, top1_motion = train(train_loader, model, criterion, optimizer, epoch, args)

        if global_rank == 0:
            writer.add_scalar('loss_joint', loss_joint.avg, global_step=epoch)
            writer.add_scalar('loss_motion', loss_motion.avg, global_step=epoch)
            writer.add_scalar('loss_sim', loss_sim.avg, global_step=epoch)
            writer.add_scalar('loss_inter', loss_inter.avg, global_step=epoch)
            writer.add_scalar('top1_joint', top1_joint.avg, global_step=epoch)
            writer.add_scalar('top1_motion', top1_motion.avg, global_step=epoch)

            if epoch % 10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=args.checkpoint_path + '/checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    losses_joint = AverageMeter('Loss Joint', ':6.3f')
    losses_motion = AverageMeter('Loss Motion', ':6.3f')
    losses_sim = AverageMeter('Loss Sim', ':6.3f')
    losses_inter = AverageMeter('Loss Inter', ':6.3f')
    losses_hand = AverageMeter('Loss_hand', ':6.3f')
    losses_hand_motion = AverageMeter('Loss_hand_motion', ':6.3f')
    losses_body = AverageMeter('Loss_body', ':6.3f')
    losses_body_motion = AverageMeter('Loss_body_motion', ':6.3f')
    top1_joint = AverageMeter('Acc Joint@1', ':6.2f')
    top1_motion = AverageMeter('Acc Motion@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses_joint, losses_motion, losses_sim, losses_hand, losses_hand_motion, losses_body, losses_inter, losses_body_motion, top1_joint, top1_motion],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch, optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_v1, input_v2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        for k, v in input_v1.items():
            input_v1[k] = v.float().cuda(non_blocking=True)
        for k, v in input_v2.items():
            input_v2[k] = v.float().cuda(non_blocking=True)


        # compute output
        output, output_motion, target, loss_sim, logits_hand, logits_body, logits_hand_motion, logits_body_motion, loss_inter_dist = model(input_v1, input_v2, self_dist=args.inter_dist)

        batch_size = output.size(0)

        # compute loss
        loss_joint = criterion(output, target)
        loss_motion = criterion(output_motion, target)
        loss_hand, loss_body, loss_hand_motion, loss_body_motion = criterion(logits_hand, target),criterion(logits_body, target),criterion(logits_hand_motion, target),criterion(logits_body_motion, target)

        loss = loss_joint + loss_motion + loss_sim + loss_hand + loss_body + loss_hand_motion + loss_body_motion + loss_inter_dist

        losses.update(loss.item(), batch_size)
        losses_joint.update(loss_joint.item(), batch_size)
        losses_motion.update(loss_motion.item(), batch_size)
        losses_sim.update(loss_sim.item(), batch_size)
        losses_hand.update(loss_hand.item(), batch_size)
        losses_hand_motion.update(loss_hand_motion.item(), batch_size)
        losses_body.update(loss_body.item(), batch_size)
        losses_body_motion.update(loss_body_motion.item(), batch_size)
        losses_sim.update(loss_sim.item(), batch_size)
        losses_inter.update(loss_inter_dist.item(), batch_size)

        # measure accuracy of model m1 and m2 individually
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1_joint, _ = accuracy(output, target, topk=(1, 5))
        acc1_motion, _ = accuracy(output_motion, target, topk=(1, 5))
        top1_joint.update(acc1_joint[0], batch_size)
        top1_motion.update(acc1_motion[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses_joint, losses_motion, losses_sim, losses_inter, top1_joint, top1_motion


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
