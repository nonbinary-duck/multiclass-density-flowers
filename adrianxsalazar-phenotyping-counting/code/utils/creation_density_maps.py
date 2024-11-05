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

import math


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
            return density

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

        print(f"original count: {np.sum(np.sum(gt))}, new: {np.sum(np.sum(density))}");
        print ('done.')
        return density


    def create_map(self, txt_format=True):
        """
        Create density maps for a dataset
        """

        json_file = open(self.dataset_path, "r");
        dataset = json.load(json_file);
        json_file.close();


        for image in dataset["images"]:
            # Get the image
            img_path = os.path.join(os.path.split(self.dataset_path)[0], "all", image["file_name"]);
            img      = plt.imread(img_path);
            imshape  = (math.floor(img.shape[0] / self.downsize_ratio), math.floor(img.shape[1] / self.downsize_ratio));


            k = np.zeros(imshape);

            minik_ratio = (1 if (self.class_count == 1) else math.ceil( math.pow(self.class_count, 0.5) ));
            minik_shape = np.multiply(imshape, math.pow(minik_ratio, -1)).astype(int);

            for c in range(self.class_count):
                # Make an image with a single 1.0 value pixel per bbox centroid
                minik = np.zeros(minik_shape);
                gt=np.loadtxt(img_path + (f"_c{c}.txt" if (self.class_count != 1) else ".txt"));

                for i in range(0,len(gt)):
                    if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                        minik[int(math.floor(gt[i][1] / (self.downsize_ratio * minik_ratio))),int(math.floor(gt[i][0] / (self.downsize_ratio * minik_ratio)))]=1.0

                # Use the gaussian kernel
                print(img_path)
                minik = self.gaussian_filter_density(minik);

                # Combine this local class in with the entire set of classes
                krow = math.floor(c / minik_ratio);
                kcol = c - (krow * minik_ratio);

                kxmin = krow * minik_shape[0];
                kxmax = (krow+1) * minik_shape[0];
                kymin = kcol * minik_shape[1];
                kymax = (kcol+1) * minik_shape[1];

                # print(f"krow {krow}, kcol {kcol}");
                # print(f"kxmin {kxmin}, kxmax {kxmax}");
                # print(f"kymin {kymin}, kymax {kymax}");
                # print(f"minik.shape {minik.shape}, minik_shape {minik_shape}");
                # print(f"k.shape {k.shape}");
                
                k[kxmin:kxmax, kymin:kymax] = minik;

            # Save completed density map
            x=img_path + ".gt.h5";
            with h5py.File(x, 'w') as hf:
                hf.create_dataset("density", data=k, compression='gzip', compression_opts=5);


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
    density_map.create_map()
