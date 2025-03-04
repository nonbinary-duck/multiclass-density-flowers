import h5py
import time
import torch
import numpy as np
from PIL import Image
import cv2
import sys

import matplotlib.pyplot as plt

# This is from https://stackoverflow.com/a/35904211
module = sys.modules[__name__]
# [0,1,2,3,4,5...n] would acheive 1:1
# [8,3,1] is more what you should use it for
module.INDEX_MAPPING = None;

def __loader_init__(_index_mapping = None):
    if _index_mapping != None:
        module.INDEX_MAPPING = _index_mapping

    # print(f"Set INDEX_MAPPING to {module.INDEX_MAPPING} {_index_mapping}")

def apply_mapping(gt_file):
    
    if (module.INDEX_MAPPING == None):
        # (target, mask)
        return (
            np.asarray(gt_file['density_map'][:,:,:]),
            np.asarray(gt_file['mask'][:,:,:])
        )
    else:
        den_shape  = gt_file['density_map'].shape
        mask_shape = gt_file['mask'].shape
        target     = np.zeros((len(module.INDEX_MAPPING), den_shape[1],  den_shape[2] ), dtype=gt_file['density_map'].dtype)
        mask       = np.zeros((len(module.INDEX_MAPPING), mask_shape[1], mask_shape[2]), dtype=gt_file['mask'].dtype)

        for i, mapping in enumerate(module.INDEX_MAPPING):
            target[i,:,:] = gt_file['density_map'][mapping,:,:]
            mask[i,:,:]   =        gt_file['mask'][mapping,:,:]
        
        return (target, mask)


def load_data_visdrone_class_8(img_path,train = True):
    while True:
        try:
            gt_path = img_path.replace('.jpg','.h5').replace('images','gt_density_map')

            # print(f">>>>> img {img_path} g {gt_path}")

            img = Image.open(img_path).convert('RGB')

            torch.cuda.synchronize()
            begin_time_test_7 = time.time()

            gt_file = h5py.File(gt_path)

            torch.cuda.synchronize()
            end_time_test_7 = time.time()
            run_time_7 = end_time_test_7 - begin_time_test_7
            # print('该循环程序运行时间7：', run_time_7)

            # mask_map = np.asarray(gt_file['mask'][()])
            # target = np.asarray(gt_file['density_map'][()][:,:,:])
            # mask = np.asarray(gt_file['mask'][()][:,:,:])
            (target, mask) = apply_mapping(gt_file);
            # target = np.asarray(gt_file['density_map'][()][0:8,:,:])
            # mask = np.asarray(gt_file['mask'][()][0:8,:,:])
            # k = np.asarray(gt_file['kpoint'][()])
            
            break
        except IOError:
            cv2.waitKey(5)

    # print(f">>>> SHAPES1 {img.size}, {mask.shape}, {target.shape}")
    img=img.copy()
    mask=mask.copy()
    mask[mask<=4]=1
    mask[mask>4]=0
    target=target.copy()
    k = 0

    # print(f">>>> SHAPES2 {img.size}, {mask.shape}, {target.shape}")

    # for i in range(8):
    #     ax = plt.subplot(5, 2, i+1);
    #     ax.imshow(mask[i, :, :]);
    #     plt.tight_layout();
    # plt.show();
    # exit(0);

    return img, target, k, mask

# Dont want this
# def load_data_dota_class_2(img_path,train = True):
    
#     while True:
#         try:
#             gt_path = img_path.replace('.png','.h5').replace('images','gt_density_map')

#             img = Image.open(img_path).convert('RGB')

#             torch.cuda.synchronize()
#             begin_time_test_7 = time.time()

#             gt_file = h5py.File(gt_path)

#             torch.cuda.synchronize()
#             end_time_test_7 = time.time()
#             run_time_7 = end_time_test_7 - begin_time_test_7
#             # print('该循环程序运行时间7：', run_time_7)

#             # mask_map = np.asarray(gt_file['mask'][()])
#             target = np.asarray(gt_file['density_map'][()][0:2,:,:])
#             mask = np.asarray(gt_file['mask'][()][0:2,:,:])
#             # k = np.asarray(gt_file['kpoint'][()])
#             break
#         except IOError:
#             cv2.waitKey(5)

#     img=img.copy()
#     mask=mask.copy()
#     mask[mask<=4]=1
#     mask[mask>4]=0
#     target=target.copy()
#     k = 0

#     return img, target, k, mask
