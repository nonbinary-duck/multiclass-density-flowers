from __future__ import division
import warnings

from Network.VisDrone_class8 import VGG
from utils import save_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import dataset
import math
from image import *

warnings.filterwarnings('ignore')
from config import args
import os
import scipy.misc
import imageio
import time
import random
import scipy.ndimage
import cv2

import pandas as pd

import datetime
import pynvml


pynvml.nvmlInit();
gpu_count = pynvml.nvmlDeviceGetCount();
gpu_handles = [ pynvml.nvmlDeviceGetHandleByIndex(gpui) for gpui in range(gpu_count) ];
gpu_power_odometer_kwh = 0.0; 
gpu_power_odometer_lastreport = time.time();

torch.cuda.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Store and save metrics as we go
metrics = { "train_loss": [], "val_mae": [], "val_mse": [] };

print(args)

from clearml import Task

# Track this in clearml and automatically rename it based on the time and date
task = Task.init(task_name=f"dsaca_DEBUG_refac_hicks_batch_{datetime.datetime.now().replace(microsecond=0).isoformat()}", project_name="flowers")

# get logger object for current task
logger = task.get_logger()

# The mapping between chanel indexes to our dataset
categories = ['leucanthemum_vulgare', 'raununculus_spp', 'heracleum_sphondylium', 'silene_dioica-latifolia', 'trifolium_repens', 'cirsium_arvense', 'stachys_sylvatica', 'rubus_fruticosus_agg'];
# categories = ['people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']


category_count = len(categories);

for arg, val in args._get_kwargs():
    task.set_parameter(f"args.{arg}", val);

task.set_parameter("gpus", [pynvml.nvmlDeviceGetName(gpuh) for gpuh in gpu_handles]);
task.set_parameter("cat_map", categories);
task.set_parameter("categories", category_count);

def feature_test(source_img, mask_gt, gt, mask, feature, save_pth, category):
    """
    The function to write qualatative examples to disk
    """
    
    imgs = [source_img]
    for i in range(feature.shape[1]):
        np.seterr(divide='ignore', invalid='ignore')
        save_data = 255 * mask_gt[0, i,:,:] / np.max(mask_gt[0, i,:,:])
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, 2)
        # save_data = cv2.putText(save_data, category[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        imgs.append(save_data)

        save_data = 255 * ((gt[0,i,:,:] + np.min(gt[0,i,:,:])) / np.max(gt[0,i,:,:]))
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
    
    for i, img in enumerate(imgs):
        fname = "";

        if (i == 0):
            fname = f"{save_pth}_INPUT.jpg"
            cv2.imwrite(fname, img);

        else:        
            # Get the cateogry
            cid  = int((i-1)/4);
            # Which image in this category are we in
            lid  = int((i-1)-(cid*4));
            # Is a GT image
            gt   = lid < 2;
            # Is a mask
            mask = (lid % 2) == 0;
            
            fname = f"{save_pth}_{cid}_{categories[cid]}_{"GT" if gt else "OUT"}_{"MASK" if mask else "COUNT" }.jpg";
            cv2.imwrite(fname, cv2.applyColorMap(img, cv2.COLORMAP_PLASMA));

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    setup_seed(0)

    # train_file = './npydata/hicks_train_small.npy'
    train_file = './npydata/hicks_train.npy'
    # train_file = './npydata/VisDrone_train.npy'
    # train_file = './npydata/VisDrone_train_small.npy'
    # val_file = './npydata/VisDrone_test.npy'
    val_file = './npydata/hicks_test.npy'

    # Load the lists of file names for validation and training
    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    # Initalise the model
    model = VGG()
    # Send it to the GPU
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()


    # MSE for density, cross entropy for masks
    mse_criterion =  nn.MSELoss(size_average=False).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    criterion = [mse_criterion, ce_criterion]

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1, last_epoch=-1)
    print(args.pre)


    # Load a prebuilt model (or continue etc.)
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

    # Store the training validation results
    best_mse  = 1e5
    best_maes = [1e5 for c in categories];
    best_mses = [1e5 for c in categories];

    # Begin training
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        end_train = time.time()
        print("train time ", end_train-start)

        # Forward and back
        train(train_list, model, criterion, optimizer, epoch, args, scheduler);

        # Run validation
        mae, mse, visi = validate(val_list, model, args);

        # Check if this epoch is the current best
        val_mae = np.mean(mae);
        is_best = val_mae < args.best_pred;
        args.best_pred = min(val_mae, args.best_pred);
        if is_best:
            best_mse = np.mean(mse)
            best_maes = mae.copy();
            best_mses = mse.copy();
            
        print(f'*\tbest MAE {args.best_pred:.3f} \tbest MSE {best_mse:.3f}');

        logger.report_scalar(title="Validation Metrics", series="mae", iteration=epoch, value=val_mae);
        logger.report_scalar(title="Validation Metrics", series="mse", iteration=epoch, value=np.mean(mse));


        # Report per category bests and currents
        for i, cat in enumerate(categories):
            print(f"*\t best {cat}_MAE {best_maes[i]:.3f} \t best {cat}_MSE {best_mses[i]:3f}");

            logger.report_scalar(title="MAE by cat", series=f"{cat}", iteration=epoch, value=mae[i]);
            logger.report_scalar(title="MSE by cat", series=f"{cat}", iteration=epoch, value=mse[i]);
        
        # Save this checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': args.best_pred,
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, args.task_id);

        # Record model outputs
        if (is_best):
            task.update_output_model(model_path=os.path.join(args.task_id, "model_best.pth"));

        end_val = time.time();
        print(f"val time {end_val - end_train}");

def train(data, model, criterion, optimizer, epoch, args, scheduler):
    """
    Parameters:
        data      : Pre-processed data (inputs, density map etc. GTs)
        model     : The model
        criterion : The loss functions
        optimizer : The optimizer
        epoch     : The current epoch
        args      : The program args
        scheduler : The learning rate scheduler
    """

    # I'm sure there is a better way to do this
    global gpu_power_odometer_kwh, gpu_power_odometer_lastreport;
    
    # Metrics
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # The data loader
    train_loader = torch.utils.data.DataLoader(
        batch_size  = args.batch_size,
        drop_last   = False,
        dataset     = dataset.listDataset_visdrone_class_8(
            data,
            args.task_id,
            shuffle = True,
            train   = True,
            seen    = model.module.seen,
            num_workers = args.workers,
            transform   = transforms.Compose(
                [
                    # transforms.Resize((512, 512)),
                    # Convert to tensor
                    transforms.ToTensor(),
                    # Normalise before we pass into the model acording to the means and std of ImageNet
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        )
    );

    # Upadte the args?! Bad
    args.lr = optimizer.param_groups[0]['lr']

    # Status
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))


    # Villanelle debug plotting code
    # for t in train_loader:
    #     mask = t[-1].cpu();
    #     for i in range(8):
    #         ax = plt.subplot(5, 2, i+1);
    #         ax.imshow(mask[0, i, :, :]);
    #         plt.tight_layout();
    #     plt.show();
    #     exit(0);

    # Put the model in train mode
    model.train();
    # Track how long the entire training epoch took
    end = time.time();

    # This is updaed in our for dataloader loop
    loss_ave = 0.0;

    # Process every batch in our train loader
    for i, (fname, img, target, kpoint, mask_map) in enumerate(train_loader):
        # Wait for all kernels in all streams on a CUDA device to complete
        torch.cuda.synchronize();

        # Mark the beginning of the epoch
        time_head = time.time();

        # Move data for this batch to the GPU
        img = img.cuda();
        mask_map = mask_map.cuda();
        # img = img * mask_map[0,:,:]
        # target = target  * mask_map[0,:,:]

        # Not sure that synchronise is needed really
        # It should be implied that the imge is moved to the GPU
        # Perhaps this is just for being certain about timings
        torch.cuda.synchronize();

        # Record the time it took to load our data onto the GPU
        data_time.update(time.time() - time_head);
        
        # Pass our data forward through through the model
        density_out_1, density_out_2, mask_out = model(img, target);

        # print( ">>>> SHOWING MODEL");
        # print(f">>>> {density_map_pre_1.shape}, {density_map_pre_2.shape}, {mask_pre.shape}");
        # ax = plt.subplot(2,2,1); ax.set_title("density_map_pre_1"); ax.imshow(density_map_pre_1.cpu().detach().numpy()[0, 0, :, :])
        # ax = plt.subplot(2,2,2); ax.set_title("density_map_pre_2"); ax.imshow(density_map_pre_2.cpu().detach().numpy()[0, 0, :, :])
        # ax = plt.subplot(2,2,3); ax.set_title("mask_pre"); ax.imshow(mask_pre.cpu().detach().numpy()[0, 0, :, :])
        # plt.show();
        # exit(0);

        # The reason for two, may be that one set is describing the probablility of not that class, and the other the probability of that class
        # For each category there are two mask outputs
        #   one representing the probability that a pixel is not classified as the category
        #   and the second representing the probability that a pixel is in the category

        # For each category, grab the two masks related to it
        # This probably keeps the tensors on the same device (on the GPU)
        mask_preds = [mask_out[:, i*2:(i+1)*2, :, :] for i in range(category_count)];
        # Also slice the GT for each category
        mask_gts   = [mask_map[:, i, :, :] for i in range(category_count)];

        # criterion = [mse_criterion, ce_criterion]
        # Fist get the MSE of the two density outputs
        loss = criterion[0](density_out_1, target) + criterion[0](density_out_2, target);
        # Cross entropy rate
        lamda = args.lamd;

        # Then compute the cross-entropy loss of the masks
        for ci in range(category_count):
            # Why do we convert mask gts to long but not the floating point model output?!
            loss += lamda * criterion[1](mask_preds[ci], mask_gts[ci].long());

        # print('mse_loss=',criterion[0](density_map_pre, target).item())

        # Update the losses average for this epoch
        losses.update(loss.item(), img.size(0));
        # Zer the gradients of the optimiser for the backward pass
        optimizer.zero_grad();
        # Propagate the loss back through the model
        loss.backward();
        optimizer.step();

        # Again this is waiting around literally for the purpose of getting timings
        torch.cuda.synchronize();
        batch_time.update(time.time() - end)
        end = time.time()

        # Print info about this epoch
        if (i % args.print_freq) == 0:
            # Also report mid-epoch stuff
            subiter = (epoch*len(train_loader)) + i;
            logger.report_scalar(title="Mid-Epoch Loss", series="losses.avg", iteration=subiter, value=losses.avg);
            logger.report_scalar(title="Mid-Epoch Loss", series="losses.val", iteration=subiter, value=losses.val);

            total_gpus_power = 0.0;
            for gpui, gpuh in enumerate(gpu_handles):
                # Returns mw
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(gpuh) / 1000;
                logger.report_scalar(title="GPU Power", series=f"cuda:{gpui}", iteration=subiter, value=gpu_power);
                total_gpus_power += gpu_power;

            # Multiply the power by seconds to interpolate usage over time (in J or ws)
            # Then /1000 to get kws, then /60^2 to get kwh
            gpu_power_odometer_kwh += ((total_gpus_power * (time.time() - gpu_power_odometer_lastreport)) / 1000) / 60**2;
            gpu_power_odometer_lastreport = time.time();
            logger.report_scalar(title="GPU Power", series=f"total", iteration=subiter, value=total_gpus_power);
            logger.report_scalar(title="GPU Power Usage", series=f"total_kwh", iteration=subiter, value=gpu_power_odometer_kwh);

            
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t",
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t",
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t",
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t");

        loss_ave += loss.item();
    loss_ave = loss_ave*1.0/len(train_loader);

    print(f"loss_ave {loss_ave}, lr {args.lr}");
    metrics['train_loss'].append(float(loss_ave));
    logger.report_scalar(title="Train Metrics", series="loss", iteration=epoch, value=loss_ave);

    
    scheduler.step();

def validate(data, model, args):
    """
    Parameters:
        data      : Pre-processed data (inputs, density map etc. GTs)
        model     : The model
        args      : The program args
    """

    print('begin validation');
    
    # The validation data loader
    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset_visdrone_class_8(
            data,
            args.task_id,
            shuffle=False,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Norm imgnet
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            ),
        )
    );

    # Put the model in evaluation mode (we're not changing weights here)
    model.eval()

    # Initalise
    mae = np.array([1.0]*len(categories))
    mse = np.array([1.0]*len(categories))
    visi = []

    # Loop over every batch in the validation loader (should be batch of 1 for val suposedly)
    for i, (fname, img, target, kpoint, mask_map)  in enumerate(val_loader):
        torch.set_num_threads(args.workers);

        # Pass input image and gt mask to GPU
        img = img.cuda();
        mask_map = mask_map.cuda();
        # img = img * mask_map[0,:,:]
        # target = target  * mask_map[0,:,:]

        # Pass the image forward to the model, not updating weights
        with torch.no_grad():
            density_map_pre,_, mask_pre = model(img, target);

        # ax = plt.subplot(2,2,1); ax.set_title("density_map_pre"); ax.imshow(density_map_pre.cpu().detach().numpy()[0, 0, :, :])
        # ax = plt.subplot(2,2,2); ax.set_title("mask_pre"); ax.imshow(mask_pre.cpu().detach().numpy()[0, 0, :, :])
        # ax = plt.subplot(2,2,3); ax.set_title("mask_pre"); ax.imshow(mask_pre.cpu().detach().numpy()[0, 1, :, :])
        # plt.show();

        # ax = plt.subplot(2,2,1); ax.set_title("softmax"); ax.imshow((F.softmax(mask_pre[0, 0:2]))
        #     .cpu().detach().numpy()[0, :, :]);
            
        # ax = plt.subplot(2,2,2); ax.set_title("softmax"); ax.imshow((F.softmax(mask_pre[0, 0:2]))
        #     .cpu().detach().numpy()[1, :, :]);

        # ax = plt.subplot(2,2,3); ax.set_title("maxsoftmax[1]"); ax.imshow((torch.max(F.softmax(mask_pre[0, 0:2]), 0, keepdim=True)[1])
        #     .cpu().detach().numpy()[0, :, :]);
        # ax = plt.subplot(2,2,4); ax.set_title("maxsoftmax[0]"); ax.imshow((torch.max(F.softmax(mask_pre[0, 0:2]), 0, keepdim=True)[0])
        #     .cpu().detach().numpy()[0, :, :]);
        # plt.show();

        masks = [];
        for ic, cat in enumerate(categories):
            
            masks.append(
                torch.max(
                    F.softmax(mask_pre[0, ic*2:(ic+1)*2 ])
                    , 0, keepdim=True
                )[1] # This is important
            );
        
        # print(f">>>> {density_map_pre.shape}, {mask_pre.shape}, {masks}, {len(masks)}");
        # mask_pre = torch.cat(masks, 0)
        # print(f">>>> {density_map_pre.shape}, {mask_pre.shape}");
        # mask_pre = torch.unsqueeze(mask_pre, 0)
        # print(f">>>> {density_map_pre.shape}, {mask_pre.shape}");
        # print(">>>> EEE")
        # print(mask_pre.shape)
        
        mask_pre = torch.cat( masks, dim=0 );
        mask_pre = torch.unsqueeze(mask_pre, dim=0);

        # print(">>>> LLL")
        # print(mask_pre.shape)
        # density_map_pre = torch.mul(density_map_pre, mask_pre)

        density_map_pre[density_map_pre < 0] = 0

        # print( ">>>> SHOWING VAL MODEL OUT");
        # print(f">>>> {density_map_pre.shape}, {mask_pre.shape}");
        # ax = plt.subplot(2,1,1); ax.set_title("density_map_pre"); ax.imshow(density_map_pre.cpu().detach().numpy()[0, 0, :, :])
        # ax = plt.subplot(2,1,2); ax.set_title("mask_pre"); ax.imshow(mask_pre.cpu().detach().numpy()[0, 0, :, :])
        # plt.show();
        # exit(0);

        for idx in range(len(categories)):
            count = torch.sum(density_map_pre[:,idx,:,:]).item()
            mae[idx] +=abs(torch.sum(target[:,idx,:,:]).item()  - count)
            mse[idx] +=abs(torch.sum(target[:,idx,:,:]).item()  - count) * abs(torch.sum(target[:,idx,:,:]).item()  - count)


        if i%25 == 0:
            print(i)
            outdir = f"./vision_map/visdrone_class8_epoch_{len(metrics["train_loss"])}";
            # make dir if not exist
            if (not os.path.isdir(outdir)):
                os.mkdir(outdir);

            # imgout_cxy = img.cpu().numpy()[0,:,:,:];
            # imgout     = np.zeros((imgout_cxy.shape[1], imgout_cxy.shape[2], imgout_cxy.shape[0]));
            # imgout[:,0,0] = imgout_cxy[0,:,0];
            # imgout[0,:,0] = imgout_cxy[0,0,:];
            # imgout[0,0,:] = imgout_cxy[:,0,0];
            # plt.imshow(imgout); plt.suptitle("Input image"); plt.savefig(os.path.join(outdir, f"{i:03}_input.png"));

            # for cati, cat in enumerate(categories):
            #     imgout = density_map_pre.cpu().numpy()[0,cati,:,:];
            #     plt.imshow(imgout); plt.suptitle(f"{cat} output={np.sum(imgout):.3f}"); plt.savefig(os.path.join(outdir, f"{i:03}_{cat}_out_count.png"));

            #     imgout = target.cpu().numpy()[0,cati,:,:];
            #     plt.imshow(imgout); plt.suptitle(f"{cat} gt={np.sum(imgout):.3f}"); plt.savefig(os.path.join(outdir, f"{i:03}_{cat}_gt_count.png"));

            #     imgout = mask_pre.cpu().numpy()[0,cati,:,:];
            #     plt.imshow(imgout); plt.suptitle(f"{cat} output={np.sum(imgout):.3f}"); plt.savefig(os.path.join(outdir, f"{i:03}_{cat}_out_mask.png"));

            #     imgout = mask_map.cpu().numpy()[0,cati,:,:];
            #     plt.imshow(imgout); plt.suptitle(f"{cat} gt={np.sum(imgout):.3f}"); plt.savefig(os.path.join(outdir, f"{i:03}_{cat}_gt_mask.png"));

            # print(density_map_pre.shape)
            # print(img.shape)
            # exit(0);

            # logger.report_image(f"{categories[0]}_out_after_mask", f"val_{i}_{fname[0]}", iteration=len(metrics["train_loss"]), image=density_map_pre.data.cpu().numpy()[0, 0, :, :], max_image_history=-1);
            # logger.report_image(f"input", f"val_{i}_{fname[0]}", iteration=len(metrics["train_loss"]), image=img.data.cpu().numpy()[0, :, :, :], max_image_history=-1);
            
            # TODO: Replaced all the hard-coded directories
            # source_img = cv2.imread('./dataset/VisDrone/test_data_class8/images/{}'.format(fname[0]))
            source_img = cv2.imread('./dataset/hicks_vdlike/test_data_class8/images/{}'.format(fname[0]))
            feature_test(source_img, mask_map.data.cpu().numpy(), target.data.cpu().numpy(), mask_pre.data.cpu().numpy(),
                         density_map_pre.data.cpu().numpy(),
                         f'{outdir}/{i}', categories)

    mae = mae*1.0 / len(val_loader)
    for idx in range(len(categories)):
        mse[idx] = math.sqrt(mse[idx] / len(val_loader))

    print('\n* VisDrone_class8', '\targs.gpu_id:',args.gpu_id )
    for i, cat in enumerate(categories):
        print(f"* {cat}_MAE {mae[i]:.3f} \t best {cat}_MSE {mse[i]:3f}");

    print('* MAE {mae:.3f}  * MSE {mse:.3f}'.format(mae=np.mean(mae), mse=np.mean(mse)));

    # Save the metrics for this epoch
    metrics['val_mae'].append(np.mean(mae));
    metrics['val_mse'].append(np.mean(mse));
    met_df = pd.DataFrame(metrics);
    met_df.to_pickle( os.path.join( "metrics", "metrics.pkl" ));
    met_df.to_csv( os.path.join( "metrics", "metrics.csv" ) );

    return mae, mse, visi


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
