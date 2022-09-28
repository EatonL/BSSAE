import torch
from utils.plot import *
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import time
import logging
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dataset.my_dataset import *
from tqdm import tqdm
from model.CNN2 import CNN2Downstream
from model.backbone import transDownstream
import wandb
import random
from model.backbone import transEncoder

wandb.init(project="BSSAE", entity="guyhub")

labels_name = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4',
               'QAM16', 'QAM64', 'QPSK', 'WBFM']

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def train(loader, val_loader, model, criterion, optimizer, y_train, y_val, name, epoch, savepath):
    loss_train = 0
    loss_val = 0

    train_pre_labels = []
    train_true_labels = []
    train_oa = []
    val_pred_labels = []
    val_true_labels = []
    val_oa = []
    valClass_oadia = {}
    for index, modulation in enumerate(labels_name):
        valClass_oadia[modulation] = []

    model.train()
    with tqdm(range(len(loader)),desc=f"Epoch_{epoch}_train") as tarinbar:
        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            loss = criterion(output, y.long())

            for x in output:
                label = np.argmax(x.cpu().detach().numpy(), axis=0)
                train_pre_labels.append(label)
            for x in y:
                # label = np.argmax(x.cpu().detach().numpy(), axis=0)
                train_true_labels.append(x.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            tarinbar.update(1)
            # break

    model.eval()
    with tqdm(range(len(val_loader)), desc=f"val") as valbar:
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = criterion(pred, y.long())

            for x in pred:
                label = np.argmax(x.cpu().detach().numpy(), axis=0)
                val_pred_labels.append(label)
            for x in y:
                # label = np.argmax(x.cpu().detach().numpy(), axis=0)
                val_true_labels.append(x.cpu().detach().numpy())

            loss_val += loss.item()

            valbar.update(1)


    oa_train = accuracy_score(train_true_labels, train_pre_labels)
    oa_val = accuracy_score(val_true_labels, val_pred_labels)

    for i in np.unique(y_train):  # 计算不同信噪比的准确率
        idx1 = np.where(y_train == i)[0]
        idx2 = np.where(y_val == i)[0]
        train_oa.append(accuracy_score(np.array(train_true_labels)[idx1], np.array(train_pre_labels)[idx1]))
        val_oa.append(accuracy_score(np.array(val_true_labels)[idx2], np.array(val_pred_labels)[idx2]))

        val_true_labels_i = np.array(val_true_labels)[idx2]
        val_pred_labels_i = np.array(val_pred_labels)[idx2]
        for j in range(len(labels_name)):
            idx3 = np.where(val_true_labels_i == j)[0]
            valClass_oadia[labels_name[j]].append(accuracy_score(np.array(val_true_labels_i)[idx3],
                                                                 np.array(val_pred_labels_i)[idx3]))
    # 不同信噪比下class准确率的折线图
    accPlot(y_train, valClass_oadia, savepath, name, epoch+1)

    # 不同信噪比下准确率的折线图
    accSNRplot(y_train, train_oa, val_oa, savepath, name, epoch+1)

    ## confusion matrix
    cnf_matrix = confusion_matrix(val_true_labels, val_pred_labels,normalize='true')
    plot_confusion_matrix(cnf_matrix, savepath, name, epoch + 1, labels_name)

    return loss_train, loss_val, oa_train, oa_val


def main():
    name = 'downstreanTrain_Pretrain_lr0.1'
    batch_size = 1024
    epochs = 600
    learning_rate = 0.1
    pretrained = 1
    issavedata = 1
    checkpoint_path = r'/media/ymhj/D/zl/selfSuperived/BSSAE_result/Lsoftmax/59_SGD0.005.pth'
    filename = r'/media/ymhj/D/lyt/Paper/SAE/Downstream/dataset/RML2016.10a/RML2016.10a_dict.pkl'
    
    # random.seed(2022)
    # torch.manual_seed(2022)
    random.seed(332)
    torch.manual_seed(332)
    
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": 'SGD',
        "seed": 2022
    }
    
    ## save rusult path
    savedir = '/media/ymhj/D/zl/selfSuperived/DrownstreamResult'
    savepath = 'SiTLsoftmax_PreTrain'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = osp.join(savedir, savepath)

    ## data path and pkl data
    datadir = r'/home/ymhj/LYT/paper_reproduction/supervised_MSACross/Datasets'
    datapath = 'dataset'  # 'dataset_ss_DWT_upcoef_lowfrequence'
    datapath = osp.join(datadir, datapath)

    # --------------load dataset---------------
    if issavedata:
        X_train = np.load(osp.join(datapath, 'X_train.npy'))
        Y_train = np.load(osp.join(datapath, 'Y_train.npy'))
        X_val = np.load(osp.join(datapath, 'X_val.npy'))
        Y_val = np.load(osp.join(datapath, 'Y_val.npy'))
        X_test = np.load(osp.join(datapath, 'X_test.npy'))
        Y_test = np.load(osp.join(datapath, 'Y_test.npy'))

    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data2016(filename,data=0)
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        np.save(osp.join(datapath, 'X_train.npy'), X_train)
        np.save(osp.join(datapath, 'Y_train.npy'), Y_train)
        np.save(osp.join(datapath, 'X_val.npy'), X_val)
        np.save(osp.join(datapath, 'Y_val.npy'), Y_val)
        np.save(osp.join(datapath, 'X_test.npy'), X_test)
        np.save(osp.join(datapath, 'Y_test.npy'), Y_test)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train[:, 0]))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val[:, 0]))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    print("**********************************************")
    print('Training on: ', X_train.shape[0], '  Val on: ', X_val.shape[0])
    print('Input: ', X_train.shape)
    del X_train, X_val
    gc.collect()
    # --------------load dataset---------------

    # --------------record---------------
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(osp.join(savepath, 'results')):
        os.makedirs(osp.join(savepath, 'results'))
    if not os.path.exists(osp.join(savepath, 'weights')):
        os.makedirs(osp.join(savepath, 'weights'))
    if not os.path.exists(osp.join(savepath, f'results/{name}')):
        os.makedirs(osp.join(savepath, f'results/{name}'))

    logging.basicConfig(
        # 日志级别
        level=logging.INFO,
        # 日志格式
        # 时间、代码所在文件名、代码行号、日志级别名字、日志信息
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        # 打印日志的时间
        datefmt=f'{time.asctime(time.localtime(time.time()))}',
        # 日志文件存放的目录（目录必须存在）及日志文件名
        filename=osp.join(savepath, f'{name}.log'),
        # 打开日志文件的方式
        filemode='w'
    )
    # --------------record---------------
    encoder = transEncoder(patch_len=4,h=4,d_model=256,N=4,dropout=0.1)
    model = transDownstream(encoder=encoder, d_model=256, class_num=11)
    # model = CNN2Downstream(isLinear=False)

    if pretrained:
        print("<-- load checkpoint! --> \n")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # for k in list(state_dict.keys()):
        #     if k.startswith('encoder.'):
        #         state_dict[k[len("encoder."):]] = state_dict[k]
        #         print(k[len("encoder."):])
        #     del state_dict[k]
        for k in list(state_dict.keys()):
            if not k.startswith('encoder.'):
                del state_dict[k]
        
        model.load_state_dict(state_dict, strict=False)

        # para_model = np.array(model.state_dict()[list(model.named_parameters())[0]])
        # para_state = np.array(state_dict[list(model.named_parameters())[0]])
        # if para_model == para_state:
        #     print("load successed!")
        # else:
        #     print("load failed")
        # exit()
        # print(state_dict.keys())
        # print('\n\n\n')
        # for name,param in model.named_parameters():
        #     print(name)

    # exit()
    # print(model)
    # exit()
    torch.cuda.set_device(0)
    model.cuda()

        # --------------load weights---------------

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    # 在训练进入平台期后进行学习率衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                           verbose=False)  # for ACC

    best_acc = 0
    for epoch in range(epochs):
        loss_train, loss_val, accuracy_train, accuracy_val = train(train_dataloader, val_dataloader, model,
                                                 criterion, optimizer, Y_train[:, 1], Y_val[:, 1], name, epoch, savepath)
        scheduler.step(loss_train)
        print(
            f"Epoch [{epoch}/{epochs}]\t Loss_train: {loss_train / len(train_dataloader)}\t "
            f"Loss_val: {loss_val / len(val_dataloader)}\t Accuracy_train: {accuracy_train}\t Accuracy_val: {accuracy_val}"
        )
        logging.info(
            f"Epoch [{epoch}/{epochs}]\t Loss_train: {loss_train / len(train_dataloader)}\t "
            f"Loss_val: {loss_val / len(val_dataloader)}\t Accuracy_train: {accuracy_train}\t Accuracy_val: {accuracy_val}")
        
        # torch.save({
        #         'epoch': epoch + 1,
        #         'arch': model,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, osp.join(savepath, f'weights/{epoch + 1}_{name}.pth'))
        if accuracy_val > best_acc:
            best_acc = accuracy_val
            torch.save({
                    'epoch': epoch + 1,
                    'arch': model,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, osp.join(savepath, f'weights/downstream_best.pth'))

        wandb.log({"downstream_loss_train": loss_train / len(train_dataloader),
                   "downstream_loss_val": loss_val / len(val_dataloader),
                   "downstream_acc_train": accuracy_train,
                   "downstream_acc_val": accuracy_val
        })

        # Optional
        wandb.watch(model)

if __name__ == "__main__":
    main()
