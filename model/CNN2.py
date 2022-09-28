from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops.layers.torch as einops

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class CNN2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.ZeroPad = nn.ConstantPad1d((4, 4), 0)
        self.ConvReMaxPool_1 = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=50,
                                                       kernel_size=8, stride=1, padding=0),
                                             nn.BatchNorm1d(50),
                                             nn.ReLU(),
                                             nn.MaxPool1d(kernel_size=2, stride=2))
        self.ConvReMaxPool_2 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=50, 
                                                       kernel_size=8, stride=1, padding=0),
                                             nn.BatchNorm1d(50),
                                             nn.ReLU(),
                                             nn.MaxPool1d(kernel_size=2, stride=2))
        self.ConvDrop = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=50, 
                                                kernel_size=4, stride=1, padding=0),
                                      nn.Dropout(0.6),
                                      nn.BatchNorm1d(50),
                                      nn.ReLU(),
                                      nn.MaxPool1d(kernel_size=2, stride=2))
        # self.head = nn.Sequential(nn.Flatten(),
                                #   nn.Linear(600, 70),
                                #   nn.SELU(),
                                #   nn.Linear(70, 11),
                                #   nn.Softmax(dim=-1))
        
    def forward(self, x):
        x = self.ZeroPad(x)
        x = self.ConvReMaxPool_1(x)
        x = self.ConvReMaxPool_2(x)
        x = self.ConvDrop(x)
        # x = self.head(x)
        return x

class CNN_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = einops.Rearrange('b c w -> b (c w)')
        self.conv1 = nn.Sequential(nn.Linear(6200, 3100),
                                   nn.Linear(3100, 2048))
        self.unflatten = einops.Rearrange('b (c w) -> b c w', c=2)

    def forward(self,x):
        x = self.flatten(x)
        x = self.conv1(x)
        x = self.unflatten(x)
        return x


class CNN2Downstream(nn.Module):
    def __init__(self, in_dim=2, isLinear=False):
        super().__init__()

        self.ZeroPad = nn.ConstantPad1d((4, 4), 0)
        self.ConvReMaxPool_1 = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=50,
                                                       kernel_size=8, stride=1, padding=0),
                                             nn.BatchNorm1d(50),
                                             nn.ReLU(),
                                             nn.MaxPool1d(kernel_size=2, stride=2))
        self.ConvReMaxPool_2 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=50,
                                                       kernel_size=8, stride=1, padding=0),
                                             nn.BatchNorm1d(50),
                                             nn.ReLU(),
                                             nn.MaxPool1d(kernel_size=2, stride=2))
        self.ConvDrop = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=50,
                                                kernel_size=4, stride=1, padding=0),
                                      nn.Dropout(0.6),
                                      nn.BatchNorm1d(50),
                                      nn.ReLU(),
                                      nn.MaxPool1d(kernel_size=2, stride=2))
        if isLinear:
            freeze(self)
        self.head = nn.Sequential(nn.Flatten(),
          nn.Linear(600, 70),
          nn.SELU(),
        #########zl add################
          nn.BatchNorm1d(70),
        #########zl add################
          nn.Linear(70, 11),
          nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.ZeroPad(x)
        x = self.ConvReMaxPool_1(x)
        x = self.ConvReMaxPool_2(x)
        x = self.ConvDrop(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    net = CNN2()
    input = torch.randn(1,2,1024)
    output = net(input)
    mid = torch.randn(1, 100, 124)
    net2 = CNN_decoder()
    out = net2(mid)
    print(out.shape)
                                      
        