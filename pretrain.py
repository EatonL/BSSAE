import random
import time
import logging
from torch.utils.data import DataLoader, TensorDataset
from dataset.my_dataset import *
import os.path as osp
from dataset.my_dataset import MyData
from tqdm import tqdm
from model.sae import SAE
from model.CNN2 import CNN2, CNN_decoder
from model.backbone import transEncoder,transDecoder
import wandb
import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange

wandb.init(project="BSSAE", entity="guyhub")

labels_name = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK',
 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']


def info_nce_loss(features1, features2, batch_size, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = torch.cat([features1.view(features1.shape[0],-1),features2.view(features2.shape[0],-1)], dim=0)
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    return logits, labels

def train(loader, val_loader, model, criterion, optimizer, y_train, y_val, name, epoch, savepath):
    loss_train = 0
    loss_val = 0

    model.train()
    with tqdm(range(len(loader)),desc=f"Epoch_{epoch}_train") as tarinbar:
        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            x = x.cuda()
            y = y.cuda()

            output = model(x)
            # output = model(signal_img)
            loss = criterion(output, x.float())

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

            output = model(x)

            loss = criterion(output, x.float())

            loss_val += loss.item()

            valbar.update(1)

    return loss_train, loss_val

def train_new(loader, val_loader, model, criterion, optimizer, epoch, criterion2, batch_size,lam=0.5):
    loss_train = 0
    loss_val = 0

    model.train()
    with tqdm(range(len(loader)),desc=f"Epoch_{epoch}_train") as tarinbar:
        for step, (x, y, z) in enumerate(loader):
            optimizer.zero_grad()

            x = x[:,:2,:]
            x1 = torch.flip(x,dims=[0])
            # s = copy.deepcopy(x)
            #
            #
            # s[:-1,:,:] = x[1:,:,:]
            # s[-1,:,:] = x[0,:,:]
            # print(x1.shape)
            x = x.cuda()
            x1 = x1.cuda()

#           low, high, output = model(x[:,2:,:512],x[:,2:,512:])

            x_low, x_high, output = model(x+lam*x1, x)

#             loss = criterion(output, x[:,:2,:].float()) + 100*(criterion2(low,high)+1)
            logits, labels = info_nce_loss(x_low, x_high, batch_size)
            loss = criterion(output,x.float()) + criterion2(logits, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            tarinbar.update(1)
            # break

    model.eval()
    with tqdm(range(len(val_loader)), desc=f"val") as valbar:
        for i, (x, y, z) in enumerate(val_loader):
            x = x[:, :2, :]
            x1 = torch.flip(x, dims=[0])
            x = x.cuda()
            x1 = x1.cuda()


            # output = model(x)
#             low, high, output = model(x[:, 2:, :512], x[:, 2:, 512:])
            x_low, x_high, output = model(x+lam*x1, x)
            #             loss = criterion(output, x[:,:2,:].float()) + 100*(criterion2(low,high)+1)
            # loss = criterion(output,x.float()) + criterion2(x_low, x_high)
            logits, labels = info_nce_loss(x_low, x_high, batch_size)
            loss = criterion(output, x.float()) + criterion2(logits, labels)
            loss_val += loss.item()
            valbar.update(1)

    return loss_train, loss_val



def datasetload(datapath):
    train_dataset = MyData(datapath, "train")
    return train_dataset

def main():
    name = 'SGD0.005'
    batch_size = 64
    epochs = 1000
    learningRate = 0.01
    random.seed(2022)
    torch.manual_seed(2022)

    wandb.config = {
        "learning_rate": learningRate,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": 'SGD',
        "seed": 2022
    }

    ## save rusult path
    savedir = '/media/ymhj/D/zl/selfSuperived/BSSAE_result'
    savepath = 'pretrain2018_newSAEinfoNCE_0.2_lam05'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = osp.join(savedir, savepath)

    ## data path and pkl data
    # datadir = r'/media/ymhj/D/lyt/signalfusionresult/RML2018Dataset/'
    # datapath = 'dataset2018_obo'
    # datapath = osp.join(datadir, datapath)
    datapath = '/media/ymhj/D/lyt/signalfusionresult/RML2018Dataset/dataset2018_obo'
   
    #--------------load myDataset---------------
    totaldata = MyData(datapath,'train') +MyData(datapath,'val')

    train_size = int(len(totaldata) * 0.2)
    validate_size = int(len(totaldata) * 0.1) #len(totaldata) - train_size
    test_size = len(totaldata) - train_size - validate_size
    train_dataset, validate_dataset, test = torch.utils.data.random_split(totaldata, [train_size, validate_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
    # train_data_disorder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)

    print("**********************************************")
    # print('Training on: ', len(train_dataloader), '  Disorder_train on: ', len(train_data_disorder))
    # exit()

    #--------------load myDataset---------------

    # --------------record---------------
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(osp.join(savepath,'results')):
        os.makedirs(osp.join(savepath,'results'))
    if not os.path.exists(osp.join(savepath,'weights')):
        os.makedirs(osp.join(savepath,'weights'))
    if not os.path.exists(osp.join(savepath,f'results/{name}')):
        os.makedirs(osp.join(savepath,f'results/{name}'))

    logging.basicConfig(
        # 日志级别
        level=logging.INFO,
        # 日志格式
        # 时间、代码所在文件名、代码行号、日志级别名字、日志信息
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        # 打印日志的时间
        datefmt=f'{time.asctime(time.localtime(time.time()))}',
        # 日志文件存放的目录（目录必须存在）及日志文件名
        filename=osp.join(savepath,f'{name}.log'),
        # 打开日志文件的方式
        filemode='w'
    )
    # --------------record---------------

    # --------------train---------------
    # encoder = CNN2(in_dim=2)
    # decoder = CNN_decoder()
    post_process = Rearrange('n h (c w) -> n c (h w)', c=2)
    encoder = transEncoder(patch_len=4,h=4,d_model=256,N=4,dropout=0.1)
    decoder = transDecoder(h=4,d_model=256,d_decode=64,N=2,dropout=0.1)
    model = SAE(encoder=encoder, decoder=decoder, isTrans=True, post_process=post_process).cuda()
    # model = SAE(encoder=encoder, decoder=decoder, isTrans=False).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    # # 余弦退火函数调整学习率
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
    # # 带重启的余弦退火学习率衰减
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=, T_mult=1, eta_min=0, last_epoch=-1,
    #                                                      verbose=False)
    # 在训练进入平台期后进行学习率衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)    # for ACC

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    # criterion2 = torch.nn.MSELoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # loss_train, loss_val = train(train_dataloader, val_dataloader, model,
        #                                          criterion, optimizer, Y_train[:, 1], Y_val[:, 1], name, epoch, savepath)

        loss_train, loss_val = train_new(train_dataloader, val_dataloader, model, criterion, optimizer, epoch, criterion2, batch_size)
        scheduler.step(loss_train)  # lr decay

        print(
            f"Epoch [{epoch}/{epochs}]\t Loss_train: {loss_train / len(train_dataloader)}\t "
            f"Loss_val: {loss_val / len(val_dataloader)}\t"
        )
        logging.info(
            f"Epoch [{epoch}/{epochs}]\t Loss_train: {loss_train / len(train_dataloader)}\t "
            f"Loss_val: {loss_val / len(val_dataloader)}\t"
        )
        torch.save({
                'epoch': epoch + 1,
                'arch': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, osp.join(savepath,f'weights/{epoch + 1}_{name}.pth'),_use_new_zipfile_serialization=True)

        wandb.log({"loss_train": loss_train / len(train_dataloader),
                   "loss_val": loss_val / len(val_dataloader)
        })

        # Optional
        wandb.watch(model)
    # --------------train---------------

#
if __name__ == "__main__":
    main()
    # weightpath = r'/media/ymhj/D/lyt/signalfusionresult/supervised_MSACross/logs_ori_gaf_nonorm_ss/weights/45_SGD0.005.pth'
    # model_state = torch.load(weightpath)
    # model = model_state['arch']
    # print(model)
    # model.load_state_dict(model_state['state_dict'])