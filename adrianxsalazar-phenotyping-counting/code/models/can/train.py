import sys
import os

import warnings

from model import CANNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

import math

import pandas as pd


metrics = { "train_loss": [], "val_mae": [] };


parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')

parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')

parser.add_argument('output', metavar='VAL',
                    help='path output')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('--best', '-b', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('--initial', '-i', metavar='PRETRAINED', default=None,type=int,
                    help='path to the pretrained model')

def main():

    global args,best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size    = 12 #26
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 1000
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 4

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)

    print(f"Training images {len(train_list)}")

    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    print(f"Validation images {len(val_list)}")

    torch.cuda.manual_seed(args.seed)

    model = CANNet()

    model = model.cuda()

    # criterion = nn.MSELoss(size_average=False).cuda()
    criterion = nn.MSELoss(reduction="sum").cuda(); # (from docs): If the field size_average is set to False, the losses are instead summed for each minibatch

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.decay)

    # Try to figure out what this sigmoid thing is
    torch.autograd.detect_anomaly(check_nan=True);


    ###########
    if args.best:
        print("=> loading best checkpoint '{}'".format(args.best))

        checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))

        model.load_state_dict(checkpoint['state_dict'])

        best_prec1=validate(val_list, model, criterion)

        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))


    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    if args.initial:
        args.start_epoch=args.initial
        print(args.initial)
        

    for epoch in range(args.start_epoch, args.epochs):

        train(train_list, model, criterion, optimizer, epoch)

        prec1 = validate(val_list, model, criterion)

        is_best = prec1 < best_prec1

        best_prec1 = min(prec1, best_prec1)

        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.output,  filename='checkpoint.pth.tar')

    print ('Train process finished')


def append_metrics(mset, mae, loss):
    # Append taken from https://stackoverflow.com/a/24284680
    mset.loc[-1] = [mae, loss]  # adding a row
    mset.index = mset.index + 1  # shifting index
    mset = mset.sort_index()  # sorting by index

def calc_mae(loader, model):
    """
    Calculate the mean average error
    The model needs to be put in eval mode BEFORE running this
    """

    mae = 0

    for i,(img, target) in enumerate(loader):
        h,w = img.shape[2:4]
        h_d = math.floor(h/2)
        w_d = math.floor(w/2)
        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

        mae += abs(pred_sum-target.sum())

    mae = mae/len(loader)
    return mae;

def train(train_list, model, criterion, optimizer, epoch):

    print(f"== Epoch {epoch} start");

    # Record epoch time
    epoch_head = time.time();

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                    ]),
                       train=True,
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i,(img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img).squeeze()#[:,:,:,:]

        # Desired is bcxy, we have bxyc
        # print(target.shape)
        # target = torch.transpose(target, 1, 3);
        # # Now we have bcyx, so swap x with y
        # target = torch.transpose(target, 2, 3);
        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        # print(f"OUTPUT SHAPE {output.shape}")
        # print(f"TARGET SHAPE {target.shape}")
        loss =  criterion(output, target)#[:, :, :])
        # loss += criterion(output[:, 1, :], target[:, 1, :])#[:, :, :])
        # loss += criterion(output[:, 2, :], target[:, 2, :])#[:, :, :])

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            
    # print(f"All but MAE at {time.time() - epoch_head:.3f} seconds into epoch");
    # Append the average loss at the end of each epoch
    # Also, calculate MAE
    model.eval();
    metrics['train_loss'].append(losses.avg);
    # metrics_train.append([losses.avg, calc_mae(train_loader, model)]);

    print(f"Finished epoch {epoch}, lasting {time.time() - epoch_head:.3f} seconds");
    


def validate(val_list, model, criterion):
    print ('begin val')
    # Record timings
    val_head = time.time();

    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)
    # batch_size=args.batch_size)

    model.eval()
    mae = calc_mae(val_loader, model);

    # for i,(img, target) in enumerate(val_loader):
    #     img = img.cuda()
    #     img = Variable(img)
    #     output = model(img)#[:,:,:,:]

    #     # Desired is bcxy, we have bxyc
    #     target = torch.transpose(target, 1, 3);
    #     # Now we have bcyx, so swap x with y
    #     target = torch.transpose(target, 2, 3);
    #     target = target.type(torch.FloatTensor).cuda()
    #     target = Variable(target)

    #     loss += criterion(output, target)#[:, :, :])

    # loss /= len(val_loader);


    # print(' * MAE {mae:.3f} '
    #           .format(mae=mae));
    print(f" * MAE {mae:.3f}");

    # Save the metrics for this epoch
    metrics['val_mae'].append(float(mae));
    met_df = pd.DataFrame(metrics);
    met_df.to_pickle( os.path.join( args.output, "metrics.pkl" ));
    met_df.to_csv( os.path.join( args.output, "metrics.csv" ) );

    return mae

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

if __name__ == '__main__':
    main()
