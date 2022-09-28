import torch.nn as nn
from einops.layers.torch import Rearrange
import torch
from copy import deepcopy
from .backbone import *
from utils.myLoss import info_nce_loss
from utils.lsoftmax import LSoftmaxLinear

'''
post_process -> Rearrange('n h (c w) -> n c (h w)', c=2)
'''


class SAE(nn.Module):
    def __init__(self, encoder, decoder, post_process=None, isTrans=True, batch_size=64, fix_init_weight=False,
                 margin=1):
        super(SAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.teacher = deepcopy(self.encoder)
        self.post_process = post_process

        self.isTrans = isTrans
        # self.log_vars = nn.Parameter(torch.zeros((2)))

        self.batch_size = batch_size
        self.loss1 = nn.MSELoss()
        self.loss2 = nn.CrossEntropyLoss()
        self.margin = margin
        self.lsoftmax_linear = LSoftmaxLinear(input_dim=self.batch_size, output_dim=self.batch_size, margin=margin)
        ######### zl add #######################
        self.projectHead = nn.Sequential(
            nn.Linear(256, 256),  # 124 256
            nn.BatchNorm1d(32),  # 50
            nn.ReLU(),
            nn.Linear(256, 256)  # 256 124
        )

        self.teacherHead = deepcopy(self.projectHead)
        ######### zl add #######################

        if not fix_init_weight:
            self.apply(self._init_weights)
        self._init_teacher()

    def _init_teacher(self):
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        for param_encoder, param_teacher in zip(self.encoder.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                                 param_encoder.data * (1. - base_momentum)

    ######### zl add #######################
    def momentum_update_PH(self, base_momentum=0):
        for param_encoder, param_teacher in zip(self.projectHead.parameters(),
                                                self.teacherHead.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                                 param_encoder.data * (1. - base_momentum)

    ######### zl add #######################

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def out(self, low_freq, high_freq):
        x_low = self.encoder(low_freq)

        if self.isTrans:
            x_low = x_low[:, 1:, :]  # remove class token
        x_low_PH = self.projectHead(x_low)  ########### zl add

        with torch.no_grad():
            x_high = self.teacher(high_freq)
            if self.isTrans:
                x_high = x_high[:, 1:, :]  # remove class token
            x_high_PH = self.teacherHead(x_high)  ########### zl add

            # self.momentum_update(self.args.base_momentum)
            self.momentum_update(0)  # 0 == turn off the momentun update
            self.momentum_update_PH(0)  ########### zl add

        output = self.decoder(x_low)
        # x_low = x_low[:, 1:, :] # remove class token
        if self.isTrans:
            output = output[:, 1:, :]
        if self.post_process:
            output = self.post_process(output)
        return x_low_PH, x_high_PH, output

    def calculate_similarity(self, features1, features2):
        features1 = features1.reshape((features1.shape[0], -1))
        features2 = features2.reshape((features2.shape[0], -1))
        similarity_matrix = torch.matmul(features1, features2.T)
        labels = torch.arange(features1.shape[0]).cuda()
        return similarity_matrix, labels

    def forward(self, x_other, x):
        x_low, x_high, output = self.out(x_other, x)
        # similarity_matrix, labels = self.calculate_similarity(x_low,x_high)
        labels = torch.arange(x_low.shape[0]).cuda()
        x_lowFlat = x_low.reshape((x_low.shape[0], -1))
        x_highFlat = x_high.reshape((x_high.shape[0], -1))
        # print('x_lowFlat: ',x_lowFlat.shape)
        logits = self.lsoftmax_linear(x_lowFlat, x_highFlat.T, labels)
        # print('logits: ',logits)
        reconstruct_loss = self.loss1(output, x)
        contrast_loss = self.loss2(logits, labels)
        loss = torch.add(reconstruct_loss, contrast_loss)
        # print('re: ',reconstruct_loss,'co: ',contrast_loss)
        return loss, reconstruct_loss, contrast_loss

        # def forward(self, x_other, x):
    #     x_low, x_high, output = self.out(x_other, x)

    #     logits, labels = info_nce_loss(x_low,x_high,self.batch_size)
    #     reconstruct_loss = self.loss1(output, x)
    #     contrast_loss = self.loss2(logits, labels)
    #     loss = torch.add(reconstruct_loss, contrast_loss)
    #     return loss, reconstruct_loss, contrast_loss


class SAEDownsream(nn.Module):
    def __init__(self, classnum=11, d_input=128, N=4, d_model=256,
                 h=4, dropout=0.1, patch_len=4):
        super(SAEDownsream, self).__init__()

        self.encoder = Trans_block(patch_len=patch_len, h=h,
                                   d_model=d_model, dropout=dropout, N=N)

        '''for RML2016.a'''
        self.fc1 = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.rrange = nn.Sequential(Rearrange('b c h -> b (c h)'))
        self.out_layer = nn.Linear(330, classnum)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc2(self.fc1(x))
        out = self.rrange(x)
        return self.out_layer(out)


if __name__ == '__main__':
    encoder = transEncoder(patch_len=16, N=4, d_model=256, h=4, dropout=0.1)
    decoder = transDecoder(N=2, d_model=256, h=4, dropout=0.1, d_decode=32, isDecoder=True)
    post_process = Rearrange('n h (c w) -> n c (h w)', c=2)

    input_low = torch.randn(2, 2, 1024)
    input_high = torch.randn(2, 2, 1024)
    net = SAE(encoder=encoder, decoder=decoder, post_process=post_process)
    low, high = net(input_low, input_high)

    print(low.shape, high.shape)