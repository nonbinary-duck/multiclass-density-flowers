import torch
from torch.utils.data import Dataset
import os
import image
import random
import time
import torch
from config import args
from torchvision import datasets, transforms
import numpy as np
from PIL import Image, ImageEnhance
import numbers

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def __init_loader__(_index_mapping = None):
    """Pass on the index mapping"""
    image.__loader_init__(_index_mapping);

class listDataset_visdrone_class_8(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=args.batch_size,
                 num_workers=args.workers, ):
        if train:
            random.shuffle(root)
        # data_keys = pre_data(root, train)
        # self.pre_data = data_keys
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        torch.cuda.synchronize()
        begin_time_test_5 = time.time()

        # img = self.lines[index]['img']
        # kpoint = self.lines[index]['kpoint']
        # target = self.lines[index]['target']
        # fname = self.lines[index]['fname']
        # mask_map = self.lines[index]['mask']
        # Img_path = self.lines[index]

        Img_path = self.lines[index]
        fname = os.path.basename(Img_path)
        img, target, kpoint, mask_map = image.load_data_visdrone_class_8(Img_path, self.train)

        torch.cuda.synchronize()
        end_time_test_5 = time.time()
        run_time_5 = end_time_test_5 - begin_time_test_5
        # print('该循环程序运行时间5：', run_time_5)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                target   = np.array([ np.fliplr(c) for c in target ]);
                mask_map = np.array([ np.fliplr(c) for c in mask_map ]);
                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # kpoint = np.fliplr(kpoint)

            # if random.random() > 0.5:
            #     proportion = random.uniform(0.004, 0.015)
            #     width, height = img.size[0], img.size[1]
            #     num = int(height * width * proportion)
            #     for i in range(num):
            #         w = random.randint(0, width - 1)
            #         h = random.randint(0, height - 1)
            #         if random.randint(0, 1) == 0:
            #             img.putpixel((w, h), (0, 0, 0))
            #         else:
            #             img.putpixel((w, h), (255, 255, 255))

        torch.cuda.synchronize()
        begin_time_test_6 = time.time()


        target = target.copy()
        # kpoint = kpoint.copy()
        img = np.array(img).copy()
        # mask_map = mask_map.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.train == True:
            # img = torch.from_numpy(img).cuda()
            target = torch.from_numpy(target).cuda()
            mask_map  = torch.from_numpy(mask_map).cuda()

            width = 512
            height = 512

            if (img.shape[-2]>512) & (img.shape[-1]>512):
                crop_size_x = random.randint(0, img.shape[-1] - width)
                crop_size_y = random.randint(0, img.shape[-2] - height)

                img = img[:, crop_size_y: crop_size_y + width, crop_size_x:crop_size_x + height]
                target = target[:, crop_size_y: crop_size_y + width, crop_size_x:crop_size_x + height]
                mask_map = mask_map[:,crop_size_y: crop_size_y + width, crop_size_x:crop_size_x + height]


        torch.cuda.synchronize()
        end_time_test_6 = time.time()
        run_time_6 = end_time_test_6 - begin_time_test_6
        # print('该循环程序运行时间6：', run_time_6)


        return fname, img, target, kpoint, mask_map

# 
# class listDataset_dota_class_2(Dataset):
#     def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=args.batch_size,
#                  num_workers=args.workers, ):
#         if train:
#             random.shuffle(root)
#         # data_keys = pre_data(root, train)
#         # self.pre_data = data_keys
#         self.nSamples = len(root)
#         self.lines = root
#         self.transform = transform
#         self.train = train
#         self.shape = shape
#         self.seen = seen
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def __len__(self):
#         return self.nSamples

#     def __getitem__(self, index):
#         assert index <= len(self), 'index range error'

#         torch.cuda.synchronize()
#         begin_time_test_5 = time.time()

#         # img = self.lines[index]['img']
#         # kpoint = self.lines[index]['kpoint']
#         # target = self.lines[index]['target']
#         # fname = self.lines[index]['fname']
#         # mask_map = self.lines[index]['mask']
#         # Img_path = self.lines[index]

#         Img_path = self.lines[index]
#         fname = os.path.basename(Img_path)
#         img, target, kpoint, mask_map = load_data_dota_class_2(Img_path, self.train)

#         torch.cuda.synchronize()
#         end_time_test_5 = time.time()
#         run_time_5 = end_time_test_5 - begin_time_test_5
#         # print('该循环程序运行时间5：', run_time_5)

#         '''data augmention'''
#         if self.train == True:
#             if random.random() > 0.5:
#                 target_0 = np.fliplr(target[0])
#                 target_1 = np.fliplr(target[1])
#                 mask_map_0 = np.fliplr(mask_map[0])
#                 mask_map_1 = np.fliplr(mask_map[1])
#                 target = np.array([target_0, target_1])
#                 mask_map = np.array([mask_map_0, mask_map_1])
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)
#                 # kpoint = np.fliplr(kpoint)

#             # if random.random() > 0.5:
#             #     proportion = random.uniform(0.004, 0.015)
#             #     width, height = img.size[0], img.size[1]
#             #     num = int(height * width * proportion)
#             #     for i in range(num):
#             #         w = random.randint(0, width - 1)
#             #         h = random.randint(0, height - 1)
#             #         if random.randint(0, 1) == 0:
#             #             img.putpixel((w, h), (0, 0, 0))
#             #         else:
#             #             img.putpixel((w, h), (255, 255, 255))

#         torch.cuda.synchronize()
#         begin_time_test_6 = time.time()


#         target = target.copy()
#         # kpoint = kpoint.copy()
#         img = np.array(img).copy()
#         # mask_map = mask_map.copy()

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.train == True:
#             # img = torch.from_numpy(img).cuda()
#             target = torch.from_numpy(target).cuda()
#             mask_map  = torch.from_numpy(mask_map).cuda()

#             width = 512
#             height = 512

#             if (img.shape[-2]>512) & (img.shape[-1]>512):
#                 crop_size_x = random.randint(0, img.shape[-1] - width)
#                 crop_size_y = random.randint(0, img.shape[-2] - height)

#                 img = img[:, crop_size_y: crop_size_y + width, crop_size_x:crop_size_x + height]
#                 target = target[:, crop_size_y: crop_size_y + width, crop_size_x:crop_size_x + height]
#                 mask_map = mask_map[:,crop_size_y: crop_size_y + width, crop_size_x:crop_size_x + height]


#         torch.cuda.synchronize()
#         end_time_test_6 = time.time()
#         run_time_6 = end_time_test_6 - begin_time_test_6
#         # print('该循环程序运行时间6：', run_time_6)


#         return fname, img, target, kpoint, mask_map


