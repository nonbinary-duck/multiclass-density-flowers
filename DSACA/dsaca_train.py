from __future__ import division

from Network.VisDrone_class8 import VGG
from utils import save_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import dsaca_dataset
import math
# from image import *

from dsaca_config import args
import os
import scipy.misc
import time
import random
import scipy.ndimage
import cv2

import pandas as pd
import numpy as np


## 
## TODO: Add for-loops where "expanded loops" can be obviously replaced
## TODO: Add comments
## TODO: Rename all occurrences referring to "validation" as "testing"
## TODO: Reduce GT to a single file
##   - The goal with this is to make it compatible with what I was doing before
##   - i.e. you have three arrays containing train/val/test input images names, and the GTs are in the same dir as input
## TODO: Include image size transformer
## TODO: Add testing file
## 




torch.cuda.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Store and save metrics as we go
metrics = { "train_loss": [], "val_mae": [], "val_mse": [] };

##
## Unconfigurable constants
##

categories = ['people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor'];



print(args)



def feature_test(source_img, mask_gt, gt, mask, feature, save_pth, category):
    """
    TODO: What does this do?
    """
    imgs = [source_img]
    for i in range(feature.shape[1]):
        np.seterr(divide='ignore', invalid='ignore')
        save_data = 255 * mask_gt[0, i,:,:] / np.max(mask_gt[0, i,:,:])
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, 2)
        # save_data = cv2.putText(save_data, category[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        imgs.append(save_data)

        save_data = 255 * gt[0,i,:,:] / np.max(gt[0,i,:,:])
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, 2)
        # save_data = cv2.putText(save_data, category[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        imgs.append(save_data)

        save_data = 255 * mask[0,i,:,:] / np.max(mask[0,i,:,:])
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, 2)
        # save_data = cv2.putText(save_data, category[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        imgs.append(save_data)

        save_data = 255 * feature[0,i,:,:] / np.max(feature[0,i,:,:])
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, 2)
        # save_data = cv2.putText(save_data, category[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        imgs.append(save_data)
    img = np.hstack(imgs)
    # for idx, image in enumerate(imgs):
    #     pth = os.path.join(os.path.dirname(save_pth), '{}.jpg'.format(idx))
    #     cv2.imwrite(pth, image)
    cv2.imwrite(save_pth, img)

def setup_seed(seed):
    """
    TODO: What does this do?
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    """
    TODO: What does this do?
    """
    setup_seed(0)

    # List of validation and train files
    train_file = './npydata/VisDrone_train.npy'
    val_file = './npydata/VisDrone_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    # Initialise the model
    model = VGG()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    mse_criterion =  nn.MSELoss(size_average=False).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    criterion = [mse_criterion, ce_criterion]

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1, last_epoch=-1)
    print(args.pre)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args.start_epoch = checkpoint['epoch']
            args.best_pred =  checkpoint['best_prec1']
            #rate_model.load_state_dict(checkpoint['rate_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    torch.set_num_threads(args.workers)

    print(args.best_pred)

    if not os.path.exists(args.task_id):
        os.makedirs(args.task_id)

    best_mse = 1e5

    best_people_mae = 1e5
    best_bicycle_mae = 1e5
    best_car_mae = 1e5
    best_van_mae = 1e5
    best_truck_mae = 1e5
    best_tricycle_mae = 1e5
    best_bus_mae = 1e5
    best_motor_mae = 1e5

    best_people_mse = 1e5
    best_bicycle_mse = 1e5
    best_car_mse = 1e5
    best_van_mse = 1e5
    best_truck_mse = 1e5
    best_tricycle_mse = 1e5
    best_bus_mse = 1e5
    best_motor_mse = 1e5

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        adjust_learning_rate(optimizer, epoch)

        end_train = time.time()
        print("train time ", end_train-start)

        if epoch <= args.max_epoch:
            # train(train_pre_load, model, criterion, optimizer, epoch, args,scheduler )
            train(train_list, model, criterion, optimizer, epoch, args,scheduler )

        #prec1, visi = validate(test_pre_load, model, args)
        mae, mse, visi = validate(val_list, model, args)

        prec1 = np.mean(mae)
        is_best = prec1 < args.best_pred
        args.best_pred = min(prec1, args.best_pred)
        if is_best:
            best_mse = np.mean(mse)
            best_people_mae = mae[0]
            best_bicycle_mae = mae[1]
            best_car_mae = mae[2]
            best_van_mae = mae[3]
            best_truck_mae = mae[4]
            best_tricycle_mae = mae[5]
            best_bus_mae = mae[6]
            best_motor_mae = mae[7]

            best_people_mse = mse[0]
            best_bicycle_mse = mse[1]
            best_car_mse = mse[2]
            best_van_mse = mse[3]
            best_truck_mse = mse[4]
            best_tricycle_mse = mse[5]
            best_bus_mse = mse[6]
            best_motor_mse = mse[7]


        print('*\tbest MAE {mae:.3f} \tbest MSE {mse:.3f}'.format(mae=args.best_pred, mse=best_mse))
        print('*\tbest people_MAE {people_mae:.3f} \tbest people_MSE {people_mse:.3f}'.format(people_mae=best_people_mae,people_mse=best_people_mse))
        print('*\tbest bicycle_MAE {bicycle_mae:.3f} \tbest bicycle_MSE {bicycle_mse:.3f}'.format(bicycle_mae=best_bicycle_mae,bicycle_mse=best_bicycle_mse))
        print('*\tbest car_MAE {car_mae:.3f} \tbest car_MSE {car_mse:.3f}'.format(car_mae=best_car_mae,car_mse=best_car_mse))
        print('*\tbest van_MAE {van_mae:.3f} \tbest van_MSE {van_mse:.3f}'.format(van_mae=best_van_mae,van_mse=best_van_mse))
        print('*\tbest truck_MAE {truck_mae:.3f} \tbest truck_MSE {truck_mse:.3f}'.format(truck_mae=best_truck_mae,truck_mse=best_truck_mse))
        print('*\tbest tricycle_MAE {tricycle_mae:.3f} \tbest tricycle_MSE {tricycle_mse:.3f}'.format(tricycle_mae=best_tricycle_mae,tricycle_mse=best_tricycle_mse))
        print('*\tbest bus_MAE {bus_mae:.3f} \tbest bus_MSE {bus_mse:.3f}'.format(bus_mae=best_bus_mae,bus_mse=best_bus_mse))
        print('*\tbest motor_MAE {motor_mae:.3f} \tbest motor_MSE {motor_mse:.3f}'.format(motor_mae=best_motor_mae,motor_mse=best_motor_mse))


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': args.best_pred,
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, args.task_id)
        end_val = time.time()
        print("val time",end_val - end_train)


def crop(d, g):
    """
    TODO: What does this do?
    """
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]

    d1 = d[:, :, abs(int(math.floor((d_h - g_h) / 2.0))):abs(int(math.floor((d_h - g_h) / 2.0))) + g_h,
         abs(int(math.floor((d_w - g_w) / 2.0))):abs(int(math.floor((d_w - g_w) / 2.0))) + g_w]
    return d1


def choose_crop(output, target):
    """
    TODO: What does this do?
    """
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    return output, target



def gt_transform(pt2d, rate):
    """
    TODO: What does this do?
    """
    # print(pt2d.shape,rate)
    pt2d = pt2d.data.cpu().numpy()

    density = np.zeros((int(rate * pt2d.shape[0]) + 1, int(rate * pt2d.shape[1]) + 1))
    pts = np.array(list(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0])))

    # print(pts.shape,np.nonzero(pt2d)[1],np.nonzero(pt2d)[0])
    orig = np.zeros((int(rate * pt2d.shape[0]) + 1, int(rate * pt2d.shape[1]) + 1))

    for i, pt in enumerate(pts):
        #    orig = np.zeros((int(rate*pt2d.shape[0])+1,int(rate*pt2d.shape[1])+1),dtype=np.float32)
        orig[int(rate * pt[1]), int(rate * pt[0])] = 1.0
    #    print(pt)

    density += scipy.ndimage.filters.gaussian_filter(orig, 8)

    # density_map = density
    # density_map = density_map / np.max(density_map) * 255
    # density_map = density_map.astype(np.uint8)
    # density_map = cv2.applyColorMap(density_map, 2)
    # cv2.imwrite('./temp/1.jpg', density_map)

    # print(np.sum(density))
    # print(pt2d.sum(),pts.shape, orig.sum(),density.sum())
    return density

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    """
    TODO: What does this do?
    """
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset_visdrone_class_8(Pre_data, args.task_id,
                            shuffle=True,
                            transform=transforms.Compose([
                                # transforms.Resize((512, 512)),
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.module.seen,
                            num_workers=args.workers),
        batch_size=args.batch_size, drop_last=False)
    args.lr = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()
    loss_ave = 0.0

    begin_time_test_4=0

    for i, (fname, img, target, kpoint, mask_map) in enumerate(train_loader):

        torch.cuda.synchronize()
        end_time_test_4 = time.time()
        run_time_4 = end_time_test_4 - begin_time_test_4
        # print('该循环程序运行时间4：', run_time_4)

        torch.cuda.synchronize()
        begin_time_test_1 = time.time()

        data_time.update(time.time() - end)
        img = img.cuda()
        # mask_map = mask_map.cuda()
        # img = img * mask_map[0,:,:]
        # target = target  * mask_map[0,:,:]

        torch.cuda.synchronize()
        end_time_test_1 = time.time()
        run_time_1 = end_time_test_1 - begin_time_test_1
        # print('该循环程序运行时间1：', run_time_1)  # 该循环程序运行时间： 1.4201874732

        torch.cuda.synchronize()
        begin_time_test_2 = time.time()

        # if epoch>307:
        #     scale = random.uniform(0.8, 1.3)
        #     img = F.upsample_bilinear(img, scale_factor=scale)
        #     target = torch.from_numpy(gt_transform(target, scale)).unsqueeze(0).type(torch.FloatTensor).cuda()
        #     print(img.shape,target.shape)
        # else:
        density_map_pre_1,density_map_pre_2, mask_pre = model(img, target)

        torch.cuda.synchronize()
        end_time_test_2 = time.time()
        run_time_2 = end_time_test_2 - begin_time_test_2
        # print('该循环程序运行时间2：', run_time_2)  # 该循环程序运行时间： 1.4201874732

        torch.cuda.synchronize()
        begin_time_test_3 = time.time()

        lamda = args.lamd
        # mask_person_pre = mask_pre[0]

        mask_people_pre = mask_pre[:, 0:2, :, :]
        mask_bicycle_pre = mask_pre[:, 2:4, :, :]
        mask_car_pre = mask_pre[:, 4:6, :, :]
        mask_van_pre = mask_pre[:, 6:8, :, :]
        mask_truck_pre = mask_pre[:, 8:10, :, :]
        mask_tricycle_pre = mask_pre[:, 10:12, :, :]
        mask_bus_pre = mask_pre[:, 12:14, :, :]
        mask_motor_pre = mask_pre[:, 14:16, :, :]

        mask_people_map = torch.unsqueeze(mask_map[0, 0, :, :], 0)
        mask_bicycle_map = torch.unsqueeze(mask_map[0, 1, :, :], 0)
        mask_car_map = torch.unsqueeze(mask_map[0, 2, :, :], 0)
        mask_van_map = torch.unsqueeze(mask_map[0, 3, :, :], 0)
        mask_truck_map = torch.unsqueeze(mask_map[0, 4, :, :], 0)
        mask_tricycle_map = torch.unsqueeze(mask_map[0, 5, :, :], 0)
        mask_bus_map = torch.unsqueeze(mask_map[0, 6, :, :], 0)
        mask_motor_map = torch.unsqueeze(mask_map[0, 7, :, :], 0)
        loss = criterion[0](density_map_pre_1, target)+criterion[0](density_map_pre_2, target) \
               + lamda * criterion[1](mask_people_pre,mask_people_map.long()) \
               + lamda * criterion[1](mask_bicycle_pre,mask_bicycle_map.long()) \
               + lamda * criterion[1](mask_car_pre,mask_car_map.long()) \
               + lamda * criterion[1](mask_van_pre,mask_van_map.long()) \
               + lamda * criterion[1](mask_truck_pre, mask_truck_map.long()) \
               + lamda * criterion[1](mask_tricycle_pre, mask_tricycle_map.long()) \
               + lamda * criterion[1](mask_bus_pre, mask_bus_map.long()) \
               + lamda * criterion[1](mask_motor_pre, mask_motor_map.long())


        # print('mse_loss=',criterion[0](density_map_pre, target).item())

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()


        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()
        end_time_test_3 = time.time()
        run_time_3 = end_time_test_3 - begin_time_test_3
        # print('该循环程序运行时间3：', run_time_3)

        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.synchronize()
        begin_time_test_4 = time.time()

        if i % args.print_freq == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        loss_ave  += loss.item()
    loss_ave = loss_ave*1.0/len(train_loader)

    print(loss_ave, args.lr)
    metrics['train_loss'].append(float(loss_ave));
    
    scheduler.step()

def validate(Pre_data, model, args):
    """
    TODO: What does this do?
    """
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_visdrone_class_8(Pre_data, args.task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),)

    model.eval()

    mae = np.array([1.0]*len(categories))
    mse = np.array([1.0]*len(categories))
    visi = []

    for i, (fname, img, target, kpoint, mask_map)  in enumerate(test_loader):
        torch.set_num_threads(args.workers)

        img = img.cuda()
        # mask_map = mask_map.cuda()
        # img = img * mask_map[0,:,:]
        # target = target  * mask_map[0,:,:]

        with torch.no_grad():
            density_map_pre,_, mask_pre = model(img, target)
        mask_people = torch.max(F.softmax(mask_pre[0,0:2]), 0, keepdim=True)[1]
        mask_bicycle = torch.max(F.softmax(mask_pre[0,2:4]), 0, keepdim=True)[1]
        mask_car = torch.max(F.softmax(mask_pre[0, 4:6]), 0, keepdim=True)[1]
        mask_van = torch.max(F.softmax(mask_pre[0, 6:8]), 0, keepdim=True)[1]
        mask_truck = torch.max(F.softmax(mask_pre[0,8:10]), 0, keepdim=True)[1]
        mask_tricycle= torch.max(F.softmax(mask_pre[0,10:12]), 0, keepdim=True)[1]
        mask_bus = torch.max(F.softmax(mask_pre[0, 12:14]), 0, keepdim=True)[1]
        mask_motor = torch.max(F.softmax(mask_pre[0, 14:16]), 0, keepdim=True)[1]
        mask_pre = torch.cat((mask_people, mask_bicycle, mask_car, mask_van, mask_truck, mask_tricycle, mask_bus, mask_motor), 0)
        mask_pre = torch.unsqueeze(mask_pre, 0)
        density_map_pre = torch.mul(density_map_pre, mask_pre)

        for idx in range(len(categories)):
            count = torch.sum(density_map_pre[:,idx,:,:]).item()
            mae[idx] +=abs(torch.sum(target[:,idx,:,:]).item()  - count)
            mse[idx] +=abs(torch.sum(target[:,idx,:,:]).item()  - count) * abs(torch.sum(target[:,idx,:,:]).item()  - count)

        density_map_pre[density_map_pre < 0] = 0

        if i%50 == 0:
            print(i)
            source_img = cv2.imread('./dataset/VisDrone/test_data_class8/images/{}'.format(fname[0]))
            feature_test(source_img, mask_map.data.cpu().numpy(), target.data.cpu().numpy(), mask_pre.data.cpu().numpy(),
                         density_map_pre.data.cpu().numpy(),
                         './vision_map/VisDrone_class8/img{}.jpg'.format(str(i)), categories)

    mae = mae*1.0 / len(test_loader)
    for idx in range(len(categories)):
        mse[idx] = math.sqrt(mse[idx] / len(test_loader))

    # VisDrone_category = ['people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']
    print('\n* VisDrone_class8', '\targs.gpu_id:',args.gpu_id )
    print('* people_MAE {people_mae:.3f}  * people_MSE {people_mse:.3f}'.format(people_mae=mae[0], people_mse=mse[0]))
    print('* bicycle_MAE {bicycle_mae:.3f}  * bicycle_MSE {bicycle_mse:.3f}'.format(bicycle_mae=mae[1], bicycle_mse=mse[1]))
    print('* car_MAE {car_mae:.3f}  * car_MSE {car_mse:.3f}'.format(car_mae=mae[2], car_mse=mse[2]))
    print('* van_MAE {van_mae:.3f}  * van_MSE {van_mse:.3f}'.format(van_mae=mae[3], van_mse=mse[3]))
    print('* truck_MAE {truck_mae:.3f}  * truck_MSE {truck_mse:.3f}'.format(truck_mae=mae[4], truck_mse=mse[4]))
    print('* tricycle_MAE {tricycle_mae:.3f}  * tricycle_MSE {tricycle_mse:.3f}'.format(tricycle_mae=mae[5], tricycle_mse=mse[5]))
    print('* bus_MAE {bus_mae:.3f}  * bus_MSE {bus_mse:.3f}'.format(bus_mae=mae[6], bus_mse=mse[6]))
    print('* motor_MAE {motor_mae:.3f}  * motor_MSE {motor_mse:.3f}'.format(motor_mae=mae[7], motor_mse=mse[7]))
    print('* MAE {mae:.3f}  * MSE {mse:.3f}'.format(mae=np.mean(mae), mse=np.mean(mse)))

    # Save the metrics for this epoch
    metrics['val_mae'].append(np.mean(mae));
    metrics['val_mse'].append(np.mean(mse));
    met_df = pd.DataFrame(metrics);
    met_df.to_pickle( os.path.join( "metrics", "metrics.pkl" ));
    met_df.to_csv( os.path.join( "metrics", "metrics.csv" ) );

    return mae, mse, visi

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # if epoch > 100:
    #     args.lr = 1e-5
    # if epoch > 300:
    #     args.lr = 1e-5


    # for i in range(len(args.steps)):
    #
    #     scale = args.scales[i] if i < len(args.scales) else 1
    #
    #     if epoch >= args.steps[i]:
    #         args.lr = args.lr * scale
    #         if epoch == args.steps[i]:
    #             break
    #     else:
    #         break
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = args.lr


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
