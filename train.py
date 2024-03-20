import argparse
import os
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset.dataloader_grundium import GrundiumZStackDataset
from logger import Logger
from models import get_classifier
from utils import save_state_dict, load_state_dict


def val(epoch, model, criterion, val_loader, logger=None):
    model.eval()

    # Init confusion matrix
    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

    with torch.no_grad():  # Disable gradient calculation
        for i, (inputs, targets) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader)):
            # CUDA setting
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)  # Get prediction
            loss = criterion(output, targets)  # Calc loss

            # Calc accuracy
            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == targets).item() / len(inputs) * 100.

            # Save log history
            if logger is not None:
                logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})

            # Save confusion matrix
            for t, p in zip(targets, preds):
                confusion_mat[int(t.item())][p.item()] += 1

    # Print and write log
    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    # Print confusion matrix
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))


def train(args, epoch, model, criterion, optimizer, train_loader, logger=None, **kwargs):
    model.train()

    # For print progress
    num_progress, next_print = 0, args.print_freq

    # Init confusion matrix
    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

    for i, (inputs, targets) in enumerate(train_loader):
        # CUDA setting
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()  # Reset gradient
        output = model(inputs)  # Get prediction
        loss = criterion(output, targets)  # Calc loss
        loss.backward()  # Calc gradient
        optimizer.step()  # Update weights

        # Calc accuracy
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == targets).item() / len(inputs) * 100.

        # Save log history
        num_progress += len(inputs)
        if logger is not None:
            logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})
            logger.add_history('batch', {'loss': loss.item(), 'accuracy': acc})

        # Save confusion matrix
        for t, p in zip(targets, preds):
            confusion_mat[int(t.item())][p.item()] += 1

        # Print and write log
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, lr=kwargs["lr"], time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    # Print log and confusion matrix
    if logger is not None:
        logger(history_key='total', epoch=epoch, lr=round(optimizer.param_groups[0]['lr'], 12))
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))


def run(args):
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Model setting
    model = get_classifier(args.model, num_classes=args.num_classes, pretrained=args.pretrained)

    # Criterion (Loss Function) setting
    criterion = nn.CrossEntropyLoss()

    # Optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler setting
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Resume
    if args.resume is not None:
        load_state_dict(args.resume, model, optimizer, scheduler)

    # CUDA
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # Dataset
    train_dataset = GrundiumZStackDataset(args, mode="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = GrundiumZStackDataset(args, mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)
    test_dataset = GrundiumZStackDataset(args, mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train(args, epoch, model, criterion, optimizer, train_loader, logger=logger, lr=scheduler.get_last_lr()[0])

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(epoch, model, criterion, val_loader, logger=logger)
            save_state_dict(os.path.join(save_dir, '{}.pth'.format(epoch)), epoch, model, optimizer, scheduler)

        # Scheduler Step
        scheduler.step()

    val("TEST", model, criterion, test_loader, logger=logger)


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--pretrained', default=True, action='store_true', help='Load pretrained model.')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    # Data Arguments
    parser.add_argument('--data', default='/mnt/16742a8d-f473-47ba-bf15-144f0e9eb6d7/data/patch_data/KBSMC/zstack/Grundium_SCAN_20240122_anonymous', help='path to dataset')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    # Training Arguments
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=10000, type=int, help='print frequency')
    parser.add_argument('--print_confusion_mat', default=False, action='store_true')
    parser.add_argument('--result', default='results_classifier', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
