import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import moco.builder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change for action recogniton
from dataset import get_finetune_training_set, get_finetune_validation_set


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', '--optim', default='SGD', type=str,
                    metavar='OPTIM', help='initial optimizer SGD|AdamW')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50, 70,], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dropout', '--dropout', default=None, type=float,
                    help='dropout in fc')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)

parser.add_argument('--data-ratio', default=0.2, type=float,
                    help='ratio of training data used in semi-supervised setting')

parser.add_argument('--finetune-skeleton-representation', default='graph-based', type=str,
                    help='which skeleton-representation to use for downstream training')
parser.add_argument('--pretrain-skeleton-representation', default='graph-based', type=str,
                    help='which skeleton-representation where used for  pre-training')
parser.add_argument('--subset_name', nargs='+', help='the datasets used NMFs_CSL|SLR500|MS_ASL|WLASL')
parser.add_argument('--num_class', type=int, help='the datasets used NMFs_CSL|SLR500|MS_ASL|WLASL')
parser.add_argument('--input_size', type=int, help='input length')
parser.add_argument('--eval_step', type=int, default=5, help='eval step')
parser.add_argument('--view', type=str, default='all',help='joint|motion|all')
parser.add_argument('--inter-dist', action='store_true',
                    help='use inter distillation loss')
parser.add_argument('--save-ckpt', action='store_false', help='if you need to cancel save checkpoint')
parser.add_argument('--protocol', type=str, default='finetune', help='finetune|semi')
best_acc1 = 0


def print_config(opt, save_path):
    config = opt.__str__()
    config = list(config.split(' '))
    for i in range(len(config)):
        config[i] = config[i] + '\n'
    with open(os.path.join(save_path,'config_txt'), 'w') as f:
        f.writelines(config)
        f.close()


def load_pretrained(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        for k in list(new_state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if not k.startswith('encoder_q'):
                del new_state_dict[k]
            elif '.proj.fc' in k:
                del new_state_dict[k]
            else:
                pass

        msg = model.load_state_dict(new_state_dict, strict=False)
        print("message",msg)
        assert set(msg.missing_keys) == {"encoder_q.proj.fc.weight", "encoder_q.proj.fc.bias",
                                         "encoder_q_motion.proj.fc.weight", "encoder_q_motion.proj.fc.bias",}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model

    # training dataset
    from options import options_classification as options
    if args.finetune_dataset == 'SLR':
        opts = options.opts_SLR_cross_subject()
        opts.train_feeder_args['subset_name'] = args.subset_name
        opts.test_feeder_args['subset_name'] = args.subset_name
        opts.train_feeder_args['num_class'] = args.num_class
        opts.test_feeder_args['num_class'] = args.num_class
        opts.train_feeder_args['input_size'] = args.input_size
        opts.test_feeder_args['input_size'] = args.input_size
        opts.num_class = args.num_class
    else:
        raise ValueError('Wrong finetune_dataset:', args.finetune_dataset)

    opts.train_feeder_args['input_representation'] = args.finetune_skeleton_representation
    opts.test_feeder_args['input_representation'] = args.finetune_skeleton_representation

    print_config(args, args.checkpoint_path)
    if 'semi' in args.protocol:
        opts.train_feeder_args['data_ratio'] = args.data_ratio
    # create summary
    writer = SummaryWriter(args.checkpoint_path)

    # create model
    print("=> creating model")

    model = moco.builder.MoCo(args.finetune_skeleton_representation, opts.num_class, pretrain=False , dropout=args.dropout)
    print("options", opts.num_class, opts.train_feeder_args, opts.test_feeder_args)

    if args.pretrained:
        # init the fc layer
        model.encoder_q.proj.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.encoder_q.proj.fc.bias.data.zero_()
        model.encoder_q_motion.proj.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.encoder_q_motion.proj.fc.bias.data.zero_()

    # load from pre-trained model
    load_pretrained(model, args.pretrained)

    if args.gpu is not None:
        model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if True:
          for parm in optimizer.param_groups:
                    print ("optimize parameters lr",parm['lr'])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print(">>>Saving checkpoints:", args.save_ckpt)
    ## Data loading code

    train_dataset = get_finetune_training_set(opts)
    val_dataset = get_finetune_validation_set(opts)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,drop_last=False)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=False)


    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1, top5, losses = train(train_loader, model, criterion, optimizer, epoch, args)

        writer.add_scalar('train_top1', top1.avg, global_step=epoch)
        writer.add_scalar('train_top5', top5.avg, global_step=epoch)
        writer.add_scalar('losses', losses.avg, global_step=epoch)
        # evaluate on validation set
        if (epoch+1) % args.eval_step == 0:
            if 'MS_ASL' in args.subset_name or 'WLASL' in args.subset_name:
                acc1, acc5, acc1_instance, acc5_instance = validate(val_loader, model, criterion, args)
                writer.add_scalar('test_top1_instance', acc1_instance, global_step=epoch)
                writer.add_scalar('test_top5_instance', acc5_instance, global_step=epoch)
                print(f'test_top1_instance:{acc1_instance}, test_top5_instance:{acc5_instance}')
            else:
                acc1, acc5 = validate(val_loader, model, criterion, args)
            writer.add_scalar('test_top1', acc1, global_step=epoch)
            writer.add_scalar('test_top5', acc5, global_step=epoch)
            print(f'top1:{acc1}, top5:{acc5}')
        else:
            acc1 = 0


        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
              print("found new best accuracy:= ",acc1)
              best_acc1 = max(acc1, best_acc1)

              if args.save_ckpt:
                  save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best,filename = args.checkpoint_path + '/best_checkpoint.pth.tar' )
                  

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        for k, v in images.items():
            images[k] = v.float().cuda(non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True).long()


        # compute output
        output = model(images, view=args.view)
        loss = criterion(output, target)

        batch_size = output.size(0)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1, top5, losses

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    if 'MS_ASL' in args.subset_name or 'WLASL' in args.subset_name:
        instance_acc = class_accuracy(args.num_class)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                for k, v in images.items():
                    images[k] = v.float().cuda(non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images, view=args.view)
            loss = criterion(output, target)

            batch_size = output.shape[0]
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            if 'MS_ASL' in args.subset_name or 'WLASL' in args.subset_name:
                instance_acc.update(output, target, (5,1))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if 'MS_ASL' in args.subset_name or 'WLASL' in args.subset_name:
            top1_acc, top5_acc = instance_acc.compute_avg()
            return top1.avg, top5.avg, top1_acc, top5_acc
    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = k[len('module.'):] if k.startswith('module.') else k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for index, param_group in enumerate(optimizer.param_groups):
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


class class_accuracy:
    def __init__(self, class_num):
        self.num = class_num
        self.correct_list_top1 = [0 for i in range(self.num)]
        self.correct_list_top5 = [0 for i in range(self.num)]
        self.list = [0 for i in range(self.num)]

    def update(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(1, 1, True, True)
            _, pred_5 = output.topk(5, 1, True, True)
            for i in range(output.shape[0]):
                self.list[target[i]] += 1
                if target[i] in pred[i]:
                    self.correct_list_top1[target[i]] += 1
                if target[i] in pred_5[i]:
                    self.correct_list_top5[target[i]] += 1

    def compute_avg(self):
        import numpy as np
        self.correct_list_top1 = np.array(self.correct_list_top1, dtype=np.float32)
        self.correct_list_top5 = np.array(self.correct_list_top5, dtype=np.float32)
        self.list = np.array(self.list, dtype=np.float32)
        top5_acc = np.mean(self.correct_list_top5 / self.list)
        top1_acc = np.mean(self.correct_list_top1 / self.list)
        return top1_acc, top5_acc



    def reset(self):
        self.correct_list_top1 = [0 for i in range(self.num)]
        self.correct_list_top5 = [0 for i in range(self.num)]
        self.list = [0 for i in range(self.num)]



if __name__ == '__main__':
    main()
