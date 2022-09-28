import numpy as np
import _pickle as pickle
import h5py
import gc
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, root_dir, dat_dir):
        self.dat_path = os.path.join(root_dir, dat_dir)
        self.listPath = self.getDataList()

    def __getitem__(self, idx):
        dat_name = self.listPath[idx]
        label = int(dat_name.split('/')[-2])
        snr = int(dat_name.split('/')[-1].split('_')[-1][:-4])

        dat = np.load(dat_name)
        dat = torch.tensor(dat)
        dat = dat.type(torch.FloatTensor)
        return dat, label, snr

    def __len__(self):
        return len(self.listPath)

    def getDataList(self):
        self.dat_list = os.listdir(self.dat_path)
        a=0
        listpath = []
        for i in self.dat_list:
            path = os.path.join(self.dat_path, i)
            listpath.extend(list(map(lambda y: os.path.join(path,y), os.listdir(path))))
            a += len(os.listdir(path))
        print(a)
        print(len(listpath))
        return listpath


# RadioML2016.10a: (220000,2,128), mods*snr*1000, total 220000 samples;
# RadioML2016.10b: (1200000,2,128), mods*snr*6000, total 1200000 samples;
# RadioML2018.01a: (2555904,2,1024), mods*snr*4096, total 2555904 samples;
def load_data2016(filename,data=0):
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]
    X = []
    lbl = []
    train_idx=[]
    val_idx=[]
    np.random.seed(2022)
    a=0
    with tqdm(total=len(mods)*len(snrs), desc='mods-snrs') as bar:
        for mod in mods:
            for snr in snrs:
                Xd_data = Xd[(mod,snr)]
                X.append(Xd_data)

                for i in range(Xd[(mod,snr)].shape[0]):
                    lbl.append((mod,snr))
                if data==0:
                    train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
                    val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
                elif data==1:
                    train_idx+=list(np.random.choice(range(a*6000,(a+1)*6000), size=3600, replace=False))
                    val_idx+=list(np.random.choice(list(set(range(a*6000,(a+1)*6000))-set(train_idx)), size=1200, replace=False))
                a+=1
                bar.update(1)

    X = np.vstack(X)

    # Scramble the order between samples
    # and get the serial number of training, validation, and test sets
    n_examples=X.shape[0]
    test_idx=list(set(range(0,n_examples))-set(train_idx)-set(val_idx))

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train =X[train_idx]
    X_val=X[val_idx]
    X_test =X[test_idx]

    Y_train=np.array(list(map(lambda x: [mods.index(lbl[x][0]),lbl[x][1]],train_idx)))
    Y_val=np.array(list(map(lambda x: [mods.index(lbl[x][0]),lbl[x][1]],val_idx)))
    Y_test=np.array(list(map(lambda x: [mods.index(lbl[x][0]),lbl[x][1]],test_idx)))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_data2018(filename):
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        X = f.get(keys[0])[:]
        X = np.reshape(X,(X.shape[0],2,1024))
        Y = f.get(keys[1])[:]
        Y = np.where(Y==1)[1]
        Y = Y[:,np.newaxis]
        Z = f.get(keys[2])[:]
    del f
    gc.collect()

    train_idx=[]
    val_idx=[]
    b = 0
    np.random.seed(2022)
    with tqdm(total=24*26, desc='mods-snrs') as bar:    # 24*26  huge dataset only select -20~18snr
        for c in range(24):
            for s in range(26):
                train_idx+=list(np.random.choice(range(b*4096,(b+1)*4096), size=int(0.7*4096), replace=False))
                val_idx+=list(np.random.choice(list(set(range(b*4096,(b+1)*4096))-set(train_idx)), size=int(0.3*4096), replace=False))
                b += 1
                bar.update(1)

    # n_examples=X.shape[0]
    # test_idx=list(set(range(0,n_examples))-set(train_idx)-set(val_idx))

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    # np.random.shuffle(test_idx)

    X_train =X[train_idx]
    X_val=X[val_idx]
    # X_test =X[test_idx]

    Y_train=np.array(list(map(lambda x: [Y[x,0],Z[x,0]],train_idx)))
    Y_val=np.array(list(map(lambda x: [Y[x,0],Z[x,0]],val_idx)))
    # Y_test=np.array(list(map(lambda x: [Y[x,0],Z[x,0]],test_idx)))
    del X, Y, Z, train_idx, val_idx #, test_idx
    gc.collect()

    return X_train, Y_train, X_val, Y_val   #, X_test, Y_test


if __name__ == "__main__":
    filename = r'/media/ymhj/D/lyt/signalfusionresult/RML2018Dataset/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
    load_data2018(filename)

