import os
import os.path as osp
import array
from torch.utils.data import Dataset
import torch
import h5py
import numpy as np
import gc
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager, Lock
import pywt
from PyEMD import EMD

LOCK = Lock()

def randomidx(trainRate,valRate):
    train_idx = []
    val_idx = []
    b = 0
    np.random.seed(2022)
    with tqdm(total=24 * 26, desc='randomIndex') as bar:  # 24*26  huge dataset only select -20~30snr
        for c in range(24):
            for s in range(26):
                train_idx += list(
                    np.random.choice(range(b * 4096, (b + 1) * 4096), size=int(trainRate * 4096), replace=False))
                val_idx += list(
                    np.random.choice(list(set(range(b * 4096, (b + 1) * 4096)) - set(train_idx)), size=int(valRate * 4096),
                                     replace=False))
                b += 1
                bar.update(1)
    return train_idx,val_idx

def DWT_hstack(signal): #concat cwtmatr and frequencies
    out = []
    for i in range(signal.shape[0]):
        [cwtmatr, frequencies] = pywt.dwt(signal[i,:],'db1') #(64,) (64,)
        out.append(np.hstack((cwtmatr,frequencies)))  #(128,)
    out = np.vstack((signal,np.array(out)))
    return out

def EMDDecomposition(input,emd):
    emd.emd(input,max_imf=1)
    imfs, res = emd.get_imfs_and_residue()
    imfs = np.sum(imfs, axis=0)
    out = np.vstack((imfs,np.array(res)))
    return out

# def run(i,savepath_train,X,Y,Z,a):
def run(i):
    with LOCK:
        if not os.path.exists(osp.join(savepath_train, f'{Y[i, 0]}')):
            os.makedirs(osp.join(savepath_train, f'{Y[i, 0]}'))
        # dataX = DWT_hstack(X[i])
        dataX_i = EMDDecomposition(X[i,0],emd)
        dataX_q = EMDDecomposition(X[i,1],emd)
        dataX = np.vstack((dataX_i,dataX_q))
        np.save(osp.join(savepath_train, f'{Y[i, 0]}/{a[Y[i, 0]]}_{Z[i, 0]}.npy'), dataX)
        a[Y[i, 0]] += 1
    # print(a)

def run2(i):
    with LOCK:
        if not os.path.exists(osp.join(savepath_train, f'{Y[i, 0]}')):
            os.makedirs(osp.join(savepath_train, f'{Y[i, 0]}'))
        # dataX = DWT_hstack(X[i])
        # dataX = EMDDecomposition(X[i],emd)
        dataX_i = EMDDecomposition(X[i, 0], emd)
        dataX_q = EMDDecomposition(X[i, 1], emd)
        dataX = np.vstack((dataX_i, dataX_q))
        np.save(osp.join(savepath_train, f'{Y[i, 0]}/{b[Y[i, 0]]}_{Z[i, 0]}.npy'), dataX)
        b[Y[i, 0]] += 1
    # print(a)

def savenpy(train_idx,savepath_train,X,Y,Z):
    np.random.shuffle(train_idx)
    a = [0 for i in range(24)] # num class = 24

    pool = multiprocessing.Pool(2)
    for i in train_idx:
        pool.apply_async(func=run, args=(i,savepath_train,X,Y,Z,a))
    pool.close()
    pool.join()

def hdf52npy(filename,savepath):
    X, Y, Z = read2018(filename)

    train_idx,val_idx = randomidx()

    if not os.path.exists(osp.join(savepath,'train')):
        os.makedirs(osp.join(savepath,'train'))
    savepath_train = osp.join(savepath,'train')
    if not os.path.exists(osp.join(savepath,'val')):
        os.makedirs(osp.join(savepath,'val'))
    savepath_val = osp.join(savepath,'val')

    np.random.shuffle(train_idx)
    a = [0 for i in range(24)]  # num class = 24
    pool = multiprocessing.Pool(2)
    for i in train_idx:
        pool.apply_async(func=run, args=(i, savepath_train, X, Y, Z, a))
    pool.close()
    pool.join()

    b = [0 for i in range(24)]  # num class = 24
    pool = multiprocessing.Pool(2)
    for i in val_idx:
        pool.apply_async(func=run, args=(i, savepath_val, X, Y, Z, b))
    pool.close()
    pool.join()

    del X,Y,Z,train_idx,val_idx
    gc.collect()


def read2018(filepath):
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        X = f.get(keys[0])[:]
        X = np.reshape(X,(X.shape[0],2,1024))
        Y = f.get(keys[1])[:]
        Y = np.where(Y==1)[1]
        Y = Y[:,np.newaxis]
        Z = f.get(keys[2])[:]
    del f
    gc.collect()
    return X,Y,Z

class MyData(Dataset):
    def __init__(self, root_dir):
        self.X, self.Y, self.Z = read2018(root_dir)

    def __getitem__(self, idx):
        dat = self.X[idx]
        label = self.Y[idx]
        SNR = self.Z[idx]
        return dat, label, SNR

    def __len__(self):
        return self.Y.shape[0]

class MyDataNPY(Dataset):
    def __init__(self, root_dir, dat_dir):
        self.root_dir = root_dir
        self.dat_dir = dat_dir
        self.dat_path = os.path.join(self.root_dir, self.dat_dir)
        self.dat_list = os.listdir(self.dat_path)
        self.dat_list.sort()
        self.label = dat_dir

    def __getitem__(self, idx):
        dat_name = self.dat_list[idx]
        dat_item_path = os.path.join(self.root_dir, self.dat_dir, dat_name)
        label = int(self.label)

        offset = int(dat_name.split('_')[-1].split('.')[0])

        dat = np.load(dat_item_path)
        dat = torch.tensor(dat)
        dat = dat.type(torch.FloatTensor)
        return dat, label, offset

    def __len__(self):
        return len(self.dat_list)

if __name__ == '__main__':
    # signal = np.ones((2,128))
    # out = DWT_hstack(signal)
    # exit()
    #
    filename = r'/media/ymhj/D/lyt/signalfusionresult/RML2018Dataset/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
    savepath = "/media/ymhj/D/lyt/signalfusionresult/RML2018Dataset/dataset2018_emd"
    # hdf52npy(filename,savepath)
    X, Y, Z = read2018(filename)

    train_idx, val_idx = randomidx(trainRate=0.7,valRate=0.3)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(osp.join(savepath, 'train')):
        os.makedirs(osp.join(savepath, 'train'))
    savepath_train = osp.join(savepath, 'train')
    if not os.path.exists(osp.join(savepath, 'val')):
        os.makedirs(osp.join(savepath, 'val'))
    savepath_val = osp.join(savepath, 'val')

    emd = EMD()

    np.random.shuffle(train_idx)
    a1 = [0 for i in range(24)]  # num class = 24
    a = Manager().list(a1)
    pool = multiprocessing.Pool(30)
    # for i in train_idx:
    pool.map(func=run, iterable=train_idx)
    pool.close()
    pool.join()

    b1 = [0 for i in range(24)]  # num class = 24
    b = Manager().list(b1)
    pool = multiprocessing.Pool(30)
    # for i in val_idx:
    pool.map(func=run2, iterable=val_idx)
    pool.close()
    pool.join()

    del X, Y, Z, train_idx, val_idx
    gc.collect()