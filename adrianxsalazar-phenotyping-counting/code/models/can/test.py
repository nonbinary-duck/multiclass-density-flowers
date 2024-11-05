import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torchvision import transforms
import argparse
import json
import matplotlib
matplotlib.use('Agg');

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib import cm as c


import shutil


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('test_json', metavar='test',
                    help='path to val json')

parser.add_argument('output', metavar='VAL',
                    help='path output')

args = parser.parse_args()

def save_dictionary(dictpath_json, dictionary_data):
    dictpath_json = os.path.join(args.output, dictpath_json);
    
    json_file = open(dictpath_json, "w")
    json.dump(dictionary_data, json_file, indent=4)
    json_file.close()


vis_path = os.path.join(args.output,'visual_results');
if (os.path.isdir(vis_path)): shutil.rmtree(vis_path, True);
if (os.path.exists(vis_path)):
    raise RuntimeError(f"Output path [{vis_path}] exists but is NOT directory, fatal error");
os.mkdir(vis_path);


checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
model = CANNet()
model = model.cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred= []
gt = []

dictionary_counts={}


# Testing images
img_paths = [];
with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

# Look-up table for classes
class_lut=[];
with open( os.path.join(os.path.dirname(args.test_json), "dataset.json") , 'r') as outfile:
    class_categories = json.load(outfile)["categories"];

    # Make a sequential list of 
    for i, cat in enumerate(class_categories):
        # Don't implement sorting by id
        if (i != cat["id"]): raise RuntimeError("Categories aren't sorted by id in the dataset JSON. Sort them above this exception or when you make the dataset (or comment this line out)");
    
        # We "know" the classes in the category list are in the order used to generate the ground-truth images for model. Add them to the LUT in that order
        class_lut.append(
            # Also make the categories make me happy
            cat["name"].replace("/", "-").replace(" ", "_").lower()
        );

classes = h5py.File(os.path.join("all", img_paths[0]) + ".gt.h5")['density'].shape[2];


print(f"Detected {classes}:");
for i, cname in enumerate(class_lut):
    print(f"  {i:02d}: {cname}");


# Value of output per class (sum of all)
metric_class_val_out = [ 0 for i in class_lut ];
metric_class_val_gt  = [ 0 for i in class_lut ];
# Every instance of a count, per class
metric_class_out     = [ [] for i in class_lut ];
metric_class_gt      = [ [] for i in class_lut ];
# Every instance of a prediction (over all classes)
metric_img_out       = [];
metric_img_gt        = [];

for img_path in img_paths:
    plain_file=os.path.basename(img_path);
    img = Image.open(os.path.join("all", img_path)).convert('RGB');
    # Half (and floor) image size as that's what the data loader does
    img = img.resize( (int(img.size[0]/2), int(img.size[1]/2)), Image.BICUBIC );
    img = transform(img).cuda();
    img = img.unsqueeze(0);
    
    entire_img=Variable(img.cuda());
    entire_den=model(entire_img).detach().cpu();
    # Stored as [batch, class, x, y]
    den=np.asarray(entire_den[0]);

    
    groundtruth = h5py.File(os.path.join("all", img_path) + ".gt.h5")['density'];
    
    
    # Remove the extension and store it
    plain_file, plain_file_ext = os.path.splitext(plain_file);


    new_img_infoline = f"= For image {plain_file} =";
    print("");
    print("=" * len(new_img_infoline));
    print(new_img_infoline);
    print("=" * len(new_img_infoline));
    print(plain_file);

    # Cumulative density (on a single channel) for the whole image
    out_all_p  = np.zeros(den[0].shape);
    out_all_gt = np.zeros(groundtruth[:, :, 0].shape);

    # Save individually the channels of GT and output
    for i, cname in enumerate(class_lut + ["ALL_CLASSES"]):
        plt.figure(figsize=(16,9), dpi=150);

        # Select if we're dealing with the final metaclass
        is_agg = cname == "ALL_CLASSES";
        out_pred = den[i] if (not is_agg) else out_all_p;
        out_gt   = groundtruth[:, :, i] if (not is_agg) else out_all_gt;

        # Get counts
        count_o  = float(np.sum(out_pred));
        count_gt = float(np.sum(out_gt));

        # Record stats, build up metaclass
        if (not is_agg):
            out_all_p += den[i];
            out_all_gt += groundtruth[:, :, i];

            metric_class_val_out[i] += count_o;
            metric_class_val_gt[i]  += count_gt;

            metric_class_out[i].append(count_o);
            metric_class_gt[i].append(count_gt);
        else:
            # Add the all metaclass to total predictions
            metric_img_out.append(count_o);
            metric_img_gt.append(count_gt);

        ax = plt.subplot(1,2,1);
        ax.set_title(f"output count={count_o}");
        visden = ax.imshow(out_pred, cmap=c.plasma);
        ax.get_figure().colorbar(visden, ax=ax, location="bottom");

        ax = plt.subplot(1,2,2);
        ax.set_title(f"gt count={count_gt}");
        visden = ax.imshow(out_gt, cmap=c.plasma);
        ax.get_figure().colorbar(visden, ax=ax, location="bottom");
        
        plt.savefig(os.path.join(vis_path, plain_file)+f"_out_{i}_{cname}.png", pad_inches=0.5, bbox_inches='tight');
        plt.close();
    
        if (not is_agg): print(f"  c_{i}: predicted: {count_o:.4f}, gt: {count_gt:.4f}");

    
    # Save model input
    plt.figure(figsize=(16,9), dpi=300);
    plt.imshow(mpimg.imread( os.path.join("boxed_imgs", img_path + ".boxed.jpg")));
    plt.savefig(os.path.join(vis_path, plain_file + '_input' + plain_file_ext),bbox_inches='tight', pad_inches = 0.5, dpi=300);
    plt.close();

    sum_pred = np.sum(den);
    sum_gt   = np.sum(groundtruth);
    dictionary_counts[plain_file] = { "pred": float(sum_pred), "gt": float(sum_gt) };

    print(f"  c_ALL: predicted: {sum_pred:.4f}, gt: {sum_gt:.4f}");

    pred.append(sum_pred);
    gt.append(sum_gt);


mae = mean_absolute_error(pred,gt);
rmse = np.sqrt(mean_squared_error(pred,gt));

save_dictionary("dic_restults.json", dictionary_counts);

# # Value of output per class (sum of all)
# metric_class_val_out = [ 0 for i in class_lut ];
# metric_class_val_gt  = [ 0 for i in class_lut ];
# # Every instance of a count, per class
# metric_class_out     = [ [] for i in class_lut ];
# metric_class_gt      = [ [] for i in class_lut ];
# # Every instance of a prediction (over all classes)
# metric_img_out       = [];
# metric_img_gt        = [];
save_dictionary("metrics_data.json", {
    "class_val_out": metric_class_val_out,
    "class_val_gt":  metric_class_val_gt,
    "class_out":     metric_class_out,
    "class_gt":      metric_class_gt,
    "img_out":       metric_img_out,
    "img_gt":        metric_img_gt
});

print('MAE: ',mae);
print('RMSE: ',rmse);
results=np.array([mae,rmse]);
np.savetxt(os.path.join(args.output,"restults.txt"),results,delimiter=',');


for i, cname in enumerate(class_lut):
    mae = mean_absolute_error(metric_class_out[i],metric_class_gt[i]);
    rmse = np.power(mean_squared_error(metric_class_out[i],metric_class_gt[i]), 0.5);
    print(f"\n{i:02d} - {cname}\n  == MAE: {mae:.4f}\n  == RMSE: {rmse:.4f}\n  == count_gt: {metric_class_val_gt[i]:.4f}\n  == count_out: {metric_class_val_out[i]:.4f}");

