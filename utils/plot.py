from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.linear_model import Lars as LARS
from sklearn.model_selection import train_test_split
import pickle
import math
import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# labels_name = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
#                  'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK',
#                  'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
labels_name = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

color1 = [147/255,75/255,67/255]
color2 = [215/255,99/255,100/255]
color3 = [239/255,122/255,109/255]
color4 = [241/255,215/255,126/255]
color5 = [177/255,206/255,70/255]
color6 = [99/255,277/255,152/255]
color7 = [147/255,148/255,231/255]
color8 = [95/255,151/255,210/255]
color9 = [157/255,195/255,231/255]

def plot_confusion_matrix(cm,save,name,epoch, labels_name, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for x in range(len(labels_name)):
        for y in range(len(labels_name)):
            value = float(format('%.2f' % cm[y, x]))  # 数值处理
            plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_name))
    plt.xticks(tick_marks, labels_name, rotation=45)
    plt.yticks(tick_marks, labels_name)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(osp.join(save, f'results/{name}/{epoch}_ConfusionMatrix.png'),
                bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
    plt.close()

def accPlot(y_train, valClass_oadia, savepath, name, epoch):
    x = range(np.unique(y_train).shape[0])

    plt.plot(x, valClass_oadia[labels_name[0]], marker='1', label=u'PSK8')
    plt.plot(x, valClass_oadia[labels_name[1]], marker='2', label=u'AM-DSB')
    plt.plot(x, valClass_oadia[labels_name[2]], marker='3', label=u'AM-SSB')
    plt.plot(x, valClass_oadia[labels_name[3]], marker='4', label=u'BPSK')
    plt.plot(x, valClass_oadia[labels_name[4]], marker='s', label=u'GPFSK')
    plt.plot(x, valClass_oadia[labels_name[5]], marker='p', label=u'GFSK')
    plt.plot(x, valClass_oadia[labels_name[6]], marker='*', label=u'PAM4')
    plt.plot(x, valClass_oadia[labels_name[7]], marker='h', label=u'QAM16')
    plt.plot(x, valClass_oadia[labels_name[8]], marker='H', label=u'QAM64')
    plt.plot(x, valClass_oadia[labels_name[9]], marker='+', label=u'QPSK')
    plt.plot(x, valClass_oadia[labels_name[10]], marker='x', label=u'WBFM')
    plt.grid(linestyle='-.')

    plt.ylim(0, 1)  # 设置x轴的刻度从2到10

    plt.grid(True)
    plt.legend()  # 让图例生效
    plt.xticks(x, np.unique(y_train), rotation=1)

    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 1.0)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    # plt.subplots_adjust(bottom=0.001)

    plt.xlabel('SNR(dB)')
    plt.ylabel("Accuracy")

    plt.savefig(osp.join(savepath, f'results/{name}/ClassACCsnr_{epoch}.png'))
    plt.close()

def accSNRplot(y_train,train_oa,val_oa,savepath,name,epoch):
    x = range(np.unique(y_train).shape[0])

    plt.plot(x, train_oa, color=color1, marker='o', mec=color1, mfc='w', label=u'trainset')
    plt.plot(x, val_oa, color=color8, marker='*', ms=10, label=u'valset')
    plt.grid(linestyle='-.')
    plt.grid(True)

    plt.ylim(0, 1)
    plt.legend()  # 让图例生效
    plt.xticks(x, np.unique(y_train), rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("SNRS")  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    for a, b in zip(x, train_oa):
        plt.text(a, b, round(b, 2))
    for a, b in zip(x, val_oa):
        plt.text(a, b, round(b, 2))

    plt.savefig(osp.join(savepath, f'results/{name}/train_val_{epoch}.png'))
    plt.close()

# if __name__ == '__main__':
#     accSNRplot(y_train, train_oa, val_oa, savepath, name, epoch)
