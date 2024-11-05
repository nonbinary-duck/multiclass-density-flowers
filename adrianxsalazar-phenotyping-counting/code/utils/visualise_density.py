# importing libraries
import h5py
import PIL
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import numpy as np
import argparse

import random


def visualise_density_map(path_image):
    """
    Show density plot with matplotlib
    """
    plt.subplot(1,2,1).imshow(PIL.Image.open(path_image));
    
    gt = np.asarray( h5py.File(path_image + ".gt.h5",'r')['density'] );
    plt.subplot(1,2,2).imshow(gt,cmap=CM.plasma);
    plt.show();


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-i', metavar='density', required=True, help='The path to the density file')

    args = parser.parse_args()

    # if len(args.b) > 1:
    #     density_map=create_density_dataset(args.i, beta=args.b)
    # else:
    if (os.path.isdir(args.i)):
        print("picking random file from dir");
        
        subfiles = os.listdir(os.path.dirname(args.i));

        img_files = [];
        for file in subfiles:
            if (os.path.splitext(file)[1] == ".png"):
                img_files.append( os.path.join(args.i, file));
    
        visualise_density_map(img_files[ random.randint(0, len(img_files)) ]);
    else:
        visualise_density_map(args.i)
