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
import numpy as np

import datetime
import pynvml
import shutil


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
task = Task.init(task_name=f"dsaca_hicks_cselect_{datetime.datetime.now().replace(microsecond=0).isoformat()}", project_name="flowers")

# get logger object for current task
logger = task.get_logger()
# Increase logger limit
logger.set_default_debug_sample_history(2000);

# The mapping between chanel indexes to our dataset
# categories = ['leucanthemum_vulgare', 'raununculus_spp', 'heracleum_sphondylium', 'silene_dioica-latifolia', 'trifolium_repens', 'cirsium_arvense', 'stachys_sylvatica', 'rubus_fruticosus_agg'];
# categories = ['people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']
# categories = ["green", "white", "early-turning", "turning", "late-turning", "red"]
categories = ['leucanthemum_vulgare', 'raununculus_spp', 'heracleum_sphondylium', 'silene_dioica-latifolia', 'trifolium_repens', 'cirsium_arvense', 'stachys_sylvatica', 'rubus_fruticosus_agg', 'vicia_cracca', 'yellow_composite', 'angelica_sylvestris', 'achillea_millefolium', 'senecio_jacobaea', 'prunella_vulgaris', 'trifolium_pratense', 'lotus_spp', 'centaurea_nigra', 'vicia_sepium-sativa', 'bellis_perennis', 'symphytum_officinale', 'knautia_arvensis', 'rhinanthus_minor', 'cirsium_vulgare', 'lathyrus_pratensis', 'taraxacum_agg']
# categories = ["r_strawberry", "u_strawberry", "r_tomato", "u_tomato"]

selected_categories = ["symphytum_officinale", "leucanthemum_vulgare", "lotus_spp", "knautia_arvensis", "trifolium_repens", "trifolium_pratense", "cirsium_arvense", "taraxacum_agg", "heracleum_sphondylium", "rubus_fruticosus_agg", "yellow_composite", "cirsium_vulgare", "raununculus_spp", "senecio_jacobaea", "lathyrus_pratensis"]

# Figure out the mapping, this SHOULD create an error if it's not found
category_mapping    = [ categories.index(sc) for sc in selected_categories ]

# Completely replace the original categories
categories = selected_categories
# Tell the data loader
dataset.__init_loader__(category_mapping);

# Importance multiplies the loss for that category
# category_importance = [1.0, 1.2953929539295392, 1.4722792607802877, 1.7880299251870324, 1.9031187790311879, 1.9616963064295483, 2.3741721854304636, 2.547069271758437, 2.6144029170464904, 2.760346487006737, 2.817288801571709, 2.817288801571709, 2.902834008097166, 3.0031413612565445, 3.1412924424972615, 3.414285714285714, 3.829105473965287, 3.880920162381597, 3.9233926128590966, 4.365296803652968, 4.709359605911331, 4.764119601328903, 5.112299465240642, 7.391752577319589, 14.412060301507537]
category_importance = [1.0 for c in categories]

category_count = len(categories);

for arg, val in args._get_kwargs():
    task.set_parameter(f"args.{arg}", val);

task.set_parameter("gpus", [pynvml.nvmlDeviceGetName(gpuh) for gpuh in gpu_handles]);
task.set_parameter("cat_map", categories);
task.set_parameter("categories", category_count);

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    setup_seed(0)

    # train_file = './npydata/hicks_train_small.npy'
    train_file = './npydata/hicks_train.npy'
    # train_file = './npydata/strawds_train.npy'
    # train_file = './npydata/aoc_train.npy'
    # train_file = './npydata/VisDrone_train.npy'
    # train_file = './npydata/VisDrone_train_small.npy'
    # val_file = './npydata/VisDrone_test.npy'
    val_file = './npydata/hicks_test.npy'
    # val_file = './npydata/strawds_test.npy'
    # val_file = './npydata/aoc_val.npy'

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

        # Save checkpoint every other epoch
        if ((epoch+1) % 2 == 0):
            new_chkpnt_path = os.path.join(args.task_id, f"checkpoint_epoch_{epoch+1}_temp.pth");
            shutil.copy(os.path.join(args.task_id, "checkpoint.pth"), new_chkpnt_path);
            task.update_output_model(model_path=new_chkpnt_path, name="checkpoint", comment=f"Cats {categories}", iteration=epoch+1);
            

        # Record model outputs
        if (is_best):
            task.update_output_model(model_path=os.path.join(args.task_id, "model_best.pth"), name="model_best", comment=f"Cats {categories}", iteration=epoch+1);
            task.set_user_properties(best_model_epoch=epoch+1);

        end_val = time.time();
        print(f"val time {end_val - end_train}");


    print(" ____________________ ") # Made using cowsay
    print("< Finished training! >") # echo "Finished training!" | cowsay -f duck
    print(" -------------------- ")
    print(" \\                   ")
    print("  \\                  ")
    print("   \\ >()_            ")
    print("      (__)__ _        ")


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
    losses_mae = AverageMeter()
    losses_ce = AverageMeter()
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
        loss_mae = criterion[0](density_out_1, target) + criterion[0](density_out_2, target);
        loss = loss_mae;
        # Cross entropy rate
        lamda = args.lamd;

        total_loss_ce = 0

        # Then compute the cross-entropy loss of the masks
        for ci in range(category_count):
            # Why do we convert mask gts to long but not the floating point model output?!
            ci_loss_ce = lamda * criterion[1](mask_preds[ci], mask_gts[ci].long());
            loss += ci_loss_ce;
            total_loss_ce += ci_loss_ce;

        # print('mse_loss=',criterion[0](density_map_pre, target).item())

        # Update the losses average for this epoch
        losses.update(loss.item(), img.size(0));
        losses_mae.update(loss_mae.item(), img.size(0));
        losses_ce.update(total_loss_ce.item(), img.size(0));
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
        if (i % max(1,int(args.print_freq / args.batch_size))) == 0:
            # Also report mid-epoch stuff
            subiter  = (epoch*(len(train_loader))) + i;
            subiter *= args.batch_size;
            logger.report_scalar(title="Mid-Epoch Loss Avg", series="losses.avg", iteration=subiter, value=losses.avg);
            logger.report_scalar(title="Mid-Epoch Loss", series="losses.val", iteration=subiter, value=losses.val);
            logger.report_scalar(title="Mid-Epoch Loss Avg", series="losses_mae.avg", iteration=subiter, value=losses_mae.avg);
            logger.report_scalar(title="Mid-Epoch Loss", series="losses_mae.val", iteration=subiter, value=losses_mae.val);
            logger.report_scalar(title="Mid-Epoch Loss Avg", series="losses_ce.avg", iteration=subiter, value=losses_ce.avg);
            logger.report_scalar(title="Mid-Epoch Loss", series="losses_ce.val", iteration=subiter, value=losses_ce.val);

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
            density_map_pre,density_map_pre_2, mask_pre = model(img, target);

        masks = [];
        for ic, cat in enumerate(categories):
            
            masks.append(
                torch.max(
                    F.softmax(mask_pre[0, ic*2:(ic+1)*2 ])
                    , 0, keepdim=True
                )[1] # This is important
            );

        
        # Apply the mask
        mask_pre = torch.cat( masks, dim=0 );
        mask_pre = torch.unsqueeze(mask_pre, dim=0);

        density_map_pre2_orignp = density_map_pre_2#.detach().clone();

        density_map_pre = torch.mul(density_map_pre, mask_pre);


        for idx in range(len(categories)):
            count = torch.sum(density_map_pre[:,idx,:,:]).item()
            mae[idx] +=abs(torch.sum(target[:,idx,:,:]).item()  - count)
            mse[idx] +=abs(torch.sum(target[:,idx,:,:]).item()  - count) * abs(torch.sum(target[:,idx,:,:]).item()  - count)


        if i%20 == 0:
            density_map_pre[density_map_pre < 0] = 0;
            density_map_pre2_orignp[density_map_pre2_orignp < 0] = 0;

            print(f"Outputting samples on validation {i}")
            
            epoch = args.start_epoch + len(metrics["train_loss"]);
            # outdir = f"./vision_map/visdrone_class8_epoch_{len(metrics["train_loss"])}";
            # # make dir if not exist
            # if (not os.path.isdir(outdir)):
            #     os.mkdir(outdir);

            # Get the input image, change from (C,X,Y) to (X,Y,C)
            imgout  = img[0,:,:,:].clone().cpu().permute(1, 2, 0).numpy();
            # Scale between 0-255
            # And uint8
            imgout -= np.min(imgout);
            imgout /= np.max(imgout);
            imgout *= 255;
            logger.report_image(title=fname[0], series=f"{i}_INPUT", iteration=epoch, image=imgout);
            # fname = f"{save_pth}_{cid}_{"GT" if gt else "OUT"}_{"MASK" if mask else "COUNT" }_{categories[cid]}_{img_name}.jpg";

            def normalise_and_cmap_rgb(arr, cid, cmap = cv2.COLORMAP_PLASMA):
                # Normalize the array to 0-1 range
                min_val = np.min(arr);
                max_val = np.max(arr);

                # Select one cat
                arr = arr[0,cid,:,:];
                
                normalised_arr = (arr - min_val) / (max_val - min_val);
                # Scale to uint8
                normalised_arr = (normalised_arr * 255).astype(np.uint8);
                # Apply cmap
                normalised_arr = cv2.applyColorMap(normalised_arr, cmap);
                
                # Convert from opencv BGR to standard RGB
                return cv2.cvtColor(src=normalised_arr, code=cv2.COLOR_BGR2RGB);

            for cid, cat in enumerate(categories):
                # Out count before applied mask
                logger.report_image(title=fname[0], series=f"{cid:02d}_OUT_COUNTRAW2_{cat}_{fname[0]}", iteration=epoch,
                    image=normalise_and_cmap_rgb(density_map_pre2_orignp.cpu().numpy(), cid)
                );
                
                # Out count
                logger.report_image(title=fname[0], series=f"{cid:02d}_OUT_COUNT_{cat}_{fname[0]}", iteration=epoch,
                    image=normalise_and_cmap_rgb(density_map_pre.cpu().numpy(), cid)
                );

                # GT count
                logger.report_image(title=fname[0], series=f"{cid:02d}_GT_COUNT_{cat}_{fname[0]}", iteration=epoch,
                    image=normalise_and_cmap_rgb(target.cpu().numpy(), cid)
                );

                # Out mask
                logger.report_image(title=fname[0], series=f"{cid:02d}_OUT_MASK_{cat}_{fname[0]}", iteration=epoch,
                    image=normalise_and_cmap_rgb(mask_pre.cpu().numpy(), cid)
                );

                # GT mask
                logger.report_image(title=fname[0], series=f"{cid:02d}_GT_MASK_{cat}_{fname[0]}", iteration=epoch,
                    image=normalise_and_cmap_rgb(mask_map.cpu().numpy(), cid)
                );

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
