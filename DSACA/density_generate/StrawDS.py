# https://data.mendeley.com/datasets/z6dtfdpzz8/1
# El-Bendary, Nashwa; Elhariri, Esraa (2022), “Strawberry-DS”, Mendeley Data, V1, doi: 10.17632/z6dtfdpzz8.1

# The dataset, which has been uploaded to the Roboflow Annotate website, is arranged for a
# total of 247 original raw Strawberry images .jpg. The dataset consists of six classes: (a) Green, (b)
# White, (c) Early-Turning, (d) Turning, (e) Late-Turning, and (F) Red. Label files .txt format con-
# taining the class names (0→Green, 1→White, 2→Early-Turning, 3→Turning, 4→Late-Turning,
# and 5→Red) with associated ground truth bounding boxes were annotated into the format re-
# quired by YOLO. The dataset contains 1062 ground truth bounding box annotations



import os
import re
import cv2
import glob
import h5py
import math
import time
import numpy as np
import scipy.spatial
import scipy.io as io
from PIL import Image, ImageDraw, ImageFont
from shutil import copyfile
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

root = '../dataset/StrawDS'

test_label_pth = os.path.join(root, 'valid/labels')
test_img_pth = os.path.join(root, 'valid/images')
train_label_pth = os.path.join(root, 'train/labels')
train_img_pth = os.path.join(root, 'train/images')

test_data_images_pth = os.path.join(root, 'test_data_class8', 'images')
test_data_map_pth = os.path.join(root, 'test_data_class8', 'gt_density_map')
test_data_show_pth = os.path.join(root, 'test_data_class8', 'gt_show')
train_data_images_pth = os.path.join(root, 'train_data_class8', 'images')
train_data_map_pth = os.path.join(root, 'train_data_class8', 'gt_density_map')
train_data_show_pth = os.path.join(root, 'train_data_class8', 'gt_show')

if not os.path.exists(test_data_images_pth):
    os.makedirs(test_data_images_pth)
if not os.path.exists(test_data_map_pth):
    os.makedirs(test_data_map_pth)
if not os.path.exists(test_data_show_pth):
    os.makedirs(test_data_show_pth)
if not os.path.exists(train_data_images_pth):
    os.makedirs(train_data_images_pth)
if not os.path.exists(train_data_map_pth):
    os.makedirs(train_data_map_pth)
if not os.path.exists(train_data_show_pth):
    os.makedirs(train_data_show_pth)

def search(root, target):
    path_buf = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            path_buf += search(path, target)
        elif os.path.splitext(path)[1] == target:
            path_buf.append(path)
    return path_buf

def load_gt_bbox(filepath):
    with open(filepath) as f:
        file = f.readlines()
    gthBBs = []
    for idx, data in enumerate(file):
        # label_line = data.split(',')
        label_line = data.split(' ') # We use space for split in this dataset
        gthBBs.append([])
        for label in label_line:
            gthBBs[idx].append(label.replace('\n',''))
    return gthBBs

def find_the_num(target, category):
    for idx,name in enumerate(category):
        if str(target).find(name) >= 0:
            return idx
    return -1

def resize(input, target_size, mode='img'):
    if mode == 'img':
        rate = target_size/max(input.shape[0], input.shape[1])
        if rate<1:
            input = cv2.resize(input, (math.floor(input.shape[1]*rate), math.floor(input.shape[0]*rate)))
        return input
    elif mode == 'coordinate':
        rate = target_size/max(input[0][0], input[0][1])
        if(rate<1):
            new_x = math.floor(input[1]*rate)
            new_y = math.floor(input[2]*rate)
        else:
            new_x = input[1]
            new_y = input[2]
        return new_x, new_y
    else:
        print('Error resize mode')

def feature_test(feature, save_pth, category):
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    for i in range(feature.shape[0]):
        np.seterr(divide='ignore', invalid='ignore')
        save_data = 255 * feature[i,:,:] / np.max(feature[i,:,:])
        save_data = save_data.astype(np.uint8)
        
        save_data = cv2.applyColorMap(save_data, cv2.COLORMAP_PLASMA)
        cv2.imwrite(os.path.join(save_pth, '{}{}'.format(category[i], '.png')), save_data)

' 类别的顺序需要按照可视化来确定 '
#“无视区域”，“行人”，“人”，“自行车”，“汽车”，“货车”，“卡车”，“三轮车”，“遮阳篷三轮车”，“公共汽车”，“摩托”，“其他” '
# VisDrone_category_buf = [ 'ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
# hicks_category_buf = ['leucanthemum_vulgare', 'raununculus_spp', 'heracleum_sphondylium', 'silene_dioica-latifolia', 'trifolium_repens', 'cirsium_arvense', 'stachys_sylvatica', 'rubus_fruticosus_agg']

strawds_category_buf = ["green", "white", "early-turning", "turning", "late-turning", "red"]



kernel_size_buf = [16, 16, 32, 32, 32, 40]
color_buf = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
             (0, 125, 125), (125, 0, 125), (125, 125, 0),
             (0, 25, 25), (25, 0, 25), (25, 25, 0),
             (0, 255, 255), (255, 0, 255), (255, 255, 0),]


path_sets = [test_label_pth, train_label_pth]
img_paths=[]
for path in path_sets:
    img_paths+=search(path, '.txt')
img_paths.sort()

print('begin convert')

space_num = 0 # 记录最多能存多少图
for pth in img_paths:
    # pth='/dssg/weixu/data_wei/VisDrone/VisDrone2019-DET-train/annotations/0000007_05499_d_0000037.txt'

    starttime = time.time()

    if str(pth).find('train') > 0:
        target_pth = pth.replace('train/labels','train_data_class8/images').replace('.txt','.jpg')
    if str(pth).find('valid') > 0:
        target_pth = pth.replace('valid/labels','test_data_class8/images').replace('.txt','.jpg')

    bbox = load_gt_bbox(pth)
    img = cv2.imread(pth.replace('labels', 'images').replace('txt', 'jpg'))
    source_shape = img.shape
    img = resize(img, 2048, 'img')

    ''' mask_map_points '''
    points = [] # 多边形的顶点坐标 # Coordinates of the vertices of a polygon

    '''有效类别只有 Valid categories are only len(VisDrone_category_buf)-2  类 resemble'''
    kpoint = np.zeros((len(strawds_category_buf), img.shape[0], img.shape[1])).astype(np.uint8)
    for item in bbox:
        # #print(VisDrone_category_buf[int(item[5])])
        # if (str(hicks_category_buf[int(item[5])]).find('people')>=0) | (str(hicks_category_buf[int(item[5])]).find('pedestrian')>=0)  :
        #     # center_x = int(item[1]) + int(item[3])/2.0
        #     # center_y = int(item[0]) + int(item[2])/10.0
        #     center_x = int(item[1]) + int(item[3])/10.0
        #     center_y = int(item[0]) + int(item[2])/2.0
        #     new_x, new_y = resize((source_shape, center_x, center_y), 1024, 'coordinate')
        #     #print(source_shape, math.floor(new_x), math.floor(new_y))
        #     kpoint[int(item[5]) -1, math.floor(new_x), math.floor(new_y)] = 1
        # elif (str(hicks_category_buf[int(item[5])]).find('ignored-regions')==-1) & (str(hicks_category_buf[int(item[5])]).find('others')==-1):

        # center_x = int(item[1]) + int(item[3])/2.0
        # center_y = int(item[0]) + int(item[2])/2.0

        # 
        # For this dataset the format is [cat, centre_x_ratio, centre_y_ratio, bbx_ratio, bby_ratio]
        #
        # print(item)
        centre_y = int(float(item[1]) * img.shape[1]);
        centre_x = int(float(item[2]) * img.shape[0]);

        # new_x, new_y = resize((source_shape, centre_x, centre_y), 2048, 'coordinate')
        new_x = min(centre_x, kpoint.shape[1] - 1);
        new_y = min(centre_y, kpoint.shape[2] - 1);
        #print(source_shape, math.floor(new_x), math.floor(new_y))
        # print(item[5]);
        # int(item[0]) is the category
        kpoint[int(item[0]), math.floor(new_x), math.floor(new_y)] = 1

        # elif str(hicks_category_buf[int(item[5])]).find('ignored-regions') >= 0:
        #     top = int(item[0])
        #     left = int(item[1])
        #     bottom = int(item[0]) + int(item[2])
        #     right = int(item[1]) + int(item[3])
        #     left, bottom = resize((source_shape, bottom, left), 1024, 'coordinate')
        #     right, top = resize((source_shape, top, right), 1024, 'coordinate')
        #     points.append([ [left,top], [left,bottom], [right,bottom], [right,top] ])
        #     #print([left,top], [left,bottom], [right,bottom], [right,top])

    ''' density_map '''
    density_map = kpoint.copy().astype(np.float32)
    # density_map[1,:,:]=density_map[0,:,:]+density_map[1,:,:]#将行人和人都记作人 # Marking pedestrians and people as human beings
    # kpoint[1,:,:]=kpoint[0,:,:]+kpoint[1,:,:]#将行人和人都记作人
    # density_map[6, :, :] = density_map[6, :, :] + density_map[7, :, :] #将“三轮车，敞篷三轮车”都记作三轮车 # ‘Tricycles, open tricycles’ as tricycles.
    # kpoint[6, :, :] = kpoint[6, :, :] + kpoint[7, :, :] #将“三轮车，敞篷三轮车”都记作三轮车

    # print (density_map.shape)

    # density_map = kpoint.copy().astype(np.float32)
    for idx in range(len(strawds_category_buf)):
        density_map[idx,:,:] = gaussian_filter(density_map[idx,:,:].astype(np.float32), kernel_size_buf[idx])

    ''' mask_map '''
    mask = np.full((img.shape[0], img.shape[1]), 1).astype(np.int8)
    for item in points:
        cv2.fillPoly(mask, np.asarray([item]), 0)

    ''' density_map_test '''
    # for idx in range(kpoint.shape[0]):
    #     print(np.sum(kpoint[idx,:,:]), np.sum(density_map[idx,:,:]))
    #print(target_pth)
    cv2.imwrite(target_pth, img*mask[:,:,None])
    feature_test(density_map, target_pth.replace('images', 'gt_show').replace('.jpg', ''), strawds_category_buf)
    cv2.imwrite(os.path.join(target_pth.replace('images', 'gt_show').replace('.jpg', ''), os.path.basename(target_pth)) , np.multiply(img, mask[:,:,None]))

    # plt.figure(figsize=(16,9), dpi=300);
    # plt.imshow(img*mask[:,:,None]);
    # plt.savefig(target_pth, bbox_inches='tight', pad_inches = 0.5, dpi=300);

    

    '''
    “无视区域”，“行人”，“人”，“自行车”，“汽车”，“货车”，“卡车”，“三轮车”，“遮阳篷三轮车”，“公共汽车”，“摩托”，“其他” '
    VisDrone_category_buf = [ 'ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
    '''

    # print(np.min(kpoint[0,:,:]), np.max(kpoint[0,:,:]), np.mean(kpoint[0,:,:]), np.sum(kpoint[0,:,:]))

    distance_map = (255 * (1-kpoint[0,:,:].copy())).astype(np.uint8)
    person=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    distance_map = (255 * (1 - kpoint[1, :, :].copy())).astype(np.uint8)
    bicycle=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    distance_map = (255 * (1 - kpoint[2, :, :].copy())).astype(np.uint8)
    car=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    distance_map = (255 * (1 - kpoint[3, :, :].copy())).astype(np.uint8)
    van=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    distance_map = (255 * (1 - kpoint[4, :, :].copy())).astype(np.uint8)
    truck=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    distance_map = (255 * (1 - kpoint[5, :, :].copy())).astype(np.uint8)
    tricycle=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    # distance_map = (255 * (1 - kpoint[6, :, :].copy())).astype(np.uint8)
    # bus=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)
    # distance_map = (255 * (1 - kpoint[7, :, :].copy())).astype(np.uint8)
    # motor=cv2.distanceTransform(distance_map, cv2.DIST_L2, 5)

    spatial_mask = np.array([person, bicycle, car, van, truck, tricycle])#, bus, motor])

    distance = 10
    spatial_mask[(spatial_mask >= 0) & (spatial_mask < 1 * distance)] = 0
    spatial_mask[(spatial_mask >= 1 * distance) & (spatial_mask < 2 * distance)] = 1
    spatial_mask[(spatial_mask >= 2 * distance) & (spatial_mask < 3 * distance)] = 2
    spatial_mask[(spatial_mask >= 3 * distance) & (spatial_mask < 4 * distance)] = 3
    spatial_mask[(spatial_mask >= 4 * distance) & (spatial_mask < 5 * distance)] = 4
    spatial_mask[(spatial_mask >= 5 * distance) & (spatial_mask < 6 * distance)] = 5
    spatial_mask[(spatial_mask >= 6 * distance) & (spatial_mask < 8 * distance)] = 6
    spatial_mask[(spatial_mask >= 8 * distance) & (spatial_mask < 12 * distance)] = 7
    spatial_mask[(spatial_mask >= 12 * distance) & (spatial_mask < 18 * distance)] = 8
    spatial_mask[(spatial_mask >= 18 * distance) & (spatial_mask < 28 * distance)] = 9
    spatial_mask[(spatial_mask >= 28 * distance)] = 10

    ''' h5 save '''
    with h5py.File(target_pth.replace('images', 'gt_density_map').replace('.jpg', '.h5'), 'w', ) as hf:
        #hf['kpoint'] = kpoint
        hf.create_dataset("density_map", compression="gzip", data=density_map);
        hf.create_dataset("mask", compression="gzip", data=spatial_mask);

    endtime = time.time()
    dtime = endtime - starttime
    space_num = space_num + 1
    print(space_num, 'run_time:', dtime, pth)
    # break
print('end convert')
