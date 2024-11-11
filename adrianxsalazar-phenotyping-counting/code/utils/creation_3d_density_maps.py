# importing libraries
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
# from scipy.ndimage.filters import gaussian_filter

import scipy
import json
from matplotlib import cm as CM
#from image import *
#from model import CSRNet
# import torch
from tqdm import tqdm
import numpy as np
import argparse

import scipy.spatial
from scipy.ndimage import gaussian_filter

import warnings
import math
import cv2


class create_density_dataset():

    def __init__(self, dataset_path, class_count = 1, downsize_ratio = 8.0, beta=0.1):
        self.dataset_path=dataset_path
        self.class_count=class_count
        self.downsize_ratio=downsize_ratio
        self.beta=beta

    def gaussian_filter_density(self, gt):
        """
        Apply a gaussian filter to a density map (gt)
        """
        
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density, 0.0, 0.0;

        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
        leafsize = 3
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=4)

        print ('generate density...')
        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = 1.

            #TODO modify this to include more neighbours also the average distance
            if gt_count > 2:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*self.beta
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

            density += gaussian_filter(pt2d, sigma, mode='constant')

        d_count = np.sum(np.sum(density));
        o_count = np.sum(np.sum(gt));
        print(f"original count: {o_count}, new: {d_count}");
        print ('done.')
        return density, d_count, o_count;


    def create_map(self, txt_format=True):
        """
        Create density maps for a dataset
        """

        json_file = open(self.dataset_path, "r");
        dataset = json.load(json_file);
        json_file.close();


        for i, image in enumerate(dataset["images"]):
            progress_line = f"= BEGIN {i+1}/{len(dataset['images'])} =";
            print("\n")
            print("="*len(progress_line))
            print(progress_line)
            print("="*len(progress_line))
            

            # Get the image
            img_path = os.path.join(os.path.split(self.dataset_path)[0], "all", image["file_name"]);
            img      = plt.imread(img_path);
            # imshape  = (math.floor(img.shape[0] / self.downsize_ratio), math.floor(img.shape[1] / self.downsize_ratio));

            if (os.path.exists(img_path + ".gt.h5")):
                print(f"SKIPPING {img_path} AS THERE EXISTS ALREADY DENSITY\n\n");
                continue;


            k = np.zeros((img.shape[0], img.shape[1], self.class_count));

            total_d_count = 0.0;
            total_o_count = 0.0;

            try:
                for c in range(self.class_count):
                    # Make an image with a single 1.0 value pixel per bbox centroid
                    minik = np.zeros((math.floor(img.shape[0] / self.downsize_ratio), math.floor(img.shape[1] / self.downsize_ratio)));
                    gt=np.loadtxt(img_path + (f"_c{c}.txt" if (self.class_count != 1 or True) else ".txt"));

                    for i in range(0,len(gt)):
                        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                            minik[int(math.floor(gt[i][1] / self.downsize_ratio)),int(math.floor(gt[i][0] / self.downsize_ratio))]=1.0

                    # Use the gaussian kernel
                    print(f"{img_path}:{c}")

                    minik, d_count, o_count = self.gaussian_filter_density(minik);
                    
                    total_o_count += o_count;
                    total_d_count += d_count;

                    # Resize to full and combine
                    k[:, :, c] = cv2.resize(minik, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)/(math.pow(self.downsize_ratio, 2));

                # Save completed density map
                x=img_path + ".gt.h5";
                with h5py.File(x, 'w') as hf:
                    hf.create_dataset("density", data=k, compression='gzip', compression_opts=5);

                progress_line = f"= FINISHED. GT_COUNT {total_o_count}, NEW_COUNT {total_d_count} =";
                print("="*len(progress_line))
                print(progress_line)
                print("="*len(progress_line))
            except:
                print(f"FAILED TO PRODUCE DENSITY FOR {img_path}. MOVING ON\n\n");
        

    def visualise_density_map(self,path_image):
        """
        Show density plot with matplotlib
        """
        plt.imshow(Image.open(path_image))
        plt.show()
        gt_file = h5py.File(path_image.replace('.png','.h5'),'r')
        groundtruth = np.asarray(gt_file['density'])
        plt.imshow(groundtruth,cmap=CM.plasma)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-i',
        metavar='dataset_directory',
        required=True,
        help='the path to the directory containing the Json file');

    parser.add_argument('-c',
        metavar='class_count',
        required=False,
        default=1,
        type=int,
        help='Count of classes');
    
    parser.add_argument('-d',
        metavar='downsize_ratio',
        required=False,
        default=8.0,
        type=float,
        help='Downsizing scale');

    # parser.add_argument('-b',
    #     metavar='beta or the gaussian filter',
    #     required=False,
    #     help="Use or don't use the adaptive geometry kernel");

    args = parser.parse_args()

    # if len(args.b) > 1:
    #     density_map=create_density_dataset(args.i, beta=args.b)
    # else:
    density_map=create_density_dataset(args.i, args.c, args.d)

    # Ignore warnings from https://stackoverflow.com/a/19167903
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        density_map.create_map()
