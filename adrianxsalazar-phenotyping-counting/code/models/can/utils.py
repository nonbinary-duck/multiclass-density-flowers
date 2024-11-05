import h5py
import torch
import shutil
import numpy as np
import os

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, output_path, filename='checkpoint.pth.tar'):
    if (not os.path.isdir(output_path)):
        if (os.path.exists(output_path)):
            raise RuntimeError(f"Output path [{output_path}] exists but is NOT directory, fatal error");
        os.mkdir(output_path);

    torch.save(state, os.path.join(output_path,filename))
    if is_best:
        print("saving_best")
        torch.save(state, os.path.join(output_path,'model_best.pth.tar'))
        #shutil.copyfile(filename, os.path.join(output_path,'model_best.pth.tar'))
