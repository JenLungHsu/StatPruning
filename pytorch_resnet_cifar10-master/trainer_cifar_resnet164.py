import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet164_cifar

import random
import numpy 

model_names = sorted(name for name in resnet164_cifar.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet164_cifar.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--pruning-method', dest='pruning_method',
                    help='The pruning method you choose',
                    default='', type=str)
parser.add_argument("--warmup_length", default=10, type=int, help="Number of warmup iterations")
parser.add_argument("--random_seed", default=1, type=int, help="random seed")
best_prec1 = 0


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # Check the save_dir exists or not
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    foldname=os.path.join(args.save_dir, args.pruning_method)
    if not os.path.exists(foldname):
        os.makedirs(foldname)

    model = torch.nn.DataParallel(resnet164_cifar.__dict__[args.arch]())
    # model = resnet_cifar10.__dict__[args.arch]()
    model.cuda()

    if args.pretrained:
        if args.arch == 'resnet164_c10':
            path=os.path.join('/ssd5/Roy/pytorch_resnet_cifar10-master/pretrained_models', args.arch + '.tar')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained model")
            
        elif args.arch == 'resnet164_c100':
            path=os.path.join('/ssd5/Roy/pytorch_resnet_cifar10-master/save_resnet164_c100/best_model.th')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained model")

    if args.pruning_method:
        
        # pruning method
        if args.arch == 'resnet164_c10':

            # load pretrain model
            path=os.path.join('/ssd5/Roy/pytorch_resnet_cifar10-master/pretrained_models', args.arch + '.tar')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])

            if args.pruning_method == 'MW':

                for i in range(14, 9, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer3[i]
                del model.module.layer3[7]
                del model.module.layer3[6]
                del model.module.layer3[3]
                del model.module.layer3[2]

                for i in range(17, 8, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer2[i]
                for i in range(7, 0, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer2[i]

                for i in range(17, 13, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(12, 9, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(8, 5, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                del model.module.layer1[4]
                del model.module.layer1[1]

                print(model.module)

            elif args.pruning_method == 'KS':

                for i in range(14, 9, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer3[i]  
                
                for i in range(17, 8, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer2[i]
                del model.module.layer2[6]
                del model.module.layer2[5]

                for i in range(16, 13, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(12, 9, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(8, 5, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                del model.module.layer1[4]

                print(model.module)


        elif args.arch == 'resnet164_c100':

            path=os.path.join('/ssd5/Roy/pytorch_resnet_cifar10-master/save_resnet164_c100/best_model.th')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            
            if args.pruning_method == 'MW':

                del model.module.layer3[16]
                del model.module.layer3[4]
                del model.module.layer3[3]

                for i in range(17, 3, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer2[i]

                for i in range(17, 13, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(12, 8, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(7, 4, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(3, 1, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                print(model.module)

            elif args.pruning_method == 'KS':
                
                for i in range(17, 3, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer2[i]

                del model.module.layer1[17]
                for i in range(15, 8, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                for i in range(7, 4, -1):  # 逆向刪除，避免索引錯亂
                    del model.module.layer1[i]
                print(model.module)



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # -----------------
    # SR-init
    if args.arch == 'resnet164_c10':
        normalize = transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                        )
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), 
            download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=32, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    elif args.arch == 'resnet164_c100':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=32, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    # -----------------



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,nesterov=True)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[100, 150], last_epoch=args.start_epoch - 1)
    # ----
    import numpy as np
    def assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def cosine_lr(optimizer, args, **kwargs):
        def _lr_adjuster(epoch, iteration):
            if epoch < args.warmup_length:
                lr = _warmup_lr(args.lr, args.warmup_length, epoch)
            else:
                e = epoch - args.warmup_length
                es = args.epochs - args.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr + 0.00001

            assign_learning_rate(optimizer, lr)

            return lr

        return _lr_adjuster

    def _warmup_lr(base_lr, warmup_length, epoch):
        return base_lr * (epoch + 1) / warmup_length
    scheduler = cosine_lr(optimizer, args)
    # ----

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        path=os.path.join('/ssd5/Roy/pytorch_resnet_cifar10-master', 'save_' + args.arch , args.pruning_method , 'best_model.th')
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']  # 如果你的檔案裡有 `state_dict`，取出來
        model.load_state_dict(state_dict)

        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        scheduler(epoch, iteration=None)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        # lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, args.pruning_method, 'checkpoint.th')) 

        if is_best == True:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, args.pruning_method, 'best_model.th')) 


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, warmup_iters=10):
    """
    Run evaluation with warm-up ignored in latency measurement
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = None

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # 計時起點
            if i >= warmup_iters and i < len(val_loader) - 1:  # ← 排除最後一個 batch
                torch.cuda.synchronize()
                end = time.time()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # 計時終點並記錄時間（跳過 warm-up）
            if i >= warmup_iters and i < len(val_loader) - 1:  # ← 排除最後一個 batch
                torch.cuda.synchronize()
                batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Avg Latency per Batch (excluding first {}): {:.5f} sec'.format(
        warmup_iters, batch_time.avg))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
