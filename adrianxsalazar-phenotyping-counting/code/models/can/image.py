import random
import os
import PIL
from PIL import Image
import numpy as np
import h5py
import cv2

import math

def load_data(img_path,train = True):
    #gt_path = img_path.replace('.png','.h5').replace('images','ground_truth')
    img = Image.open( os.path.join("all", img_path)).convert('RGB')
    gt_file = h5py.File(os.path.join("all", img_path) + ".gt.h5",'r')
    target = np.asarray(gt_file['density'])

    # Half the dimensions
    target = cv2.resize(target,(math.floor(target.shape[1]/2), math.floor(target.shape[0]/2)),interpolation = cv2.INTER_CUBIC)*4
    img = img.resize((target.shape[1], target.shape[0]), PIL.Image.BICUBIC);

    # if img.size != (4032, 3024):
    #     img=img.resize((4032, 3024))
    #     target=cv2.resize(target,(4032, 3024),interpolation = cv2.INTER_CUBIC)

    if train:
        ratio = 0.5
        crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
        rdn_value = random.random()
        if rdn_value<0.25:
            dx = 0
            dy = 0
        elif rdn_value<0.5:
            dx = int(img.size[0]*ratio)
            dy = 0
        elif rdn_value<0.75:
            dx = 0
            dy = int(img.size[1]*ratio)
        else:
            dx = int(img.size[0]*ratio)
            dy = int(img.size[1]*ratio)

        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # print(target.shape)
    target = cv2.resize(target,(math.floor(target.shape[1]/8), math.floor(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    # print(target.shape)

    return img,target
