from .basic_model import *
import torch
import torch.nn as nn

class Trans_block(nn.Module):
    def __init__(self, patch_len=16, h=4, d_model=256, dropout=0.1, N=4, isDecoder=False, d_decode=64):
        super(Trans_block, self).__init__()
        
        if isDecoder:
            decoder_attn = MultiHeadedAttention(h, d_decode)
            decoder_ff = PositionwiseFeedForward(d_decode, d_decode*2, dropout)
            decoder_position = PositionalEncoding(d_decode, dropout, cls=True)
            self.embed = nn.Sequential(nn.Linear(d_model, d_decode, bias=True),
                                                    decoder_position)
            self.process = Transformer(
                Encoder(EncoderLayer(d_decode, decoder_attn, decoder_ff, dropout), N)
            )
        else:
            encoder_attn = MultiHeadedAttention(h, d_model)
            encoder_ff = PositionwiseFeedForward(d_model, d_model*2, dropout)
            
            encoder_position = PositionalEncoding(d_model, dropout, cls=True)
            self.embed = nn.Sequential(embedding(patch_len, d_model),
                                            encoder_position)
            self.process= Transformer(
                Encoder(EncoderLayer(d_model, encoder_attn, encoder_ff, dropout), N)
            )

    def forward(self, x):
        x = self.embed(x)
        x = self.process(x)
        return x

def transEncoder(patch_len, h, d_model, dropout, N):
    encoder = Trans_block(patch_len=patch_len, h=h, 
                        d_model=d_model, dropout=dropout, N=N)
    return encoder

def transDecoder(h, d_model, dropout, N, d_decode, isDecoder=True):
    decoder = Trans_block(h=h, d_model=d_model, dropout=dropout, N=N, 
                        isDecoder=isDecoder, d_decode=d_decode)
    return decoder

class transDownstream(nn.Module):
    def __init__(self, encoder, d_model=256, class_num=11):
        super(transDownstream, self).__init__()

        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(256, 70),
                  nn.SELU(),
                  nn.BatchNorm1d(70),
                  nn.Linear(70, class_num),
                  nn.Softmax(dim=-1))

    def forward(self,x):
        x = self.encoder(x)
        x = x[:, 1:, :]
        x = x.mean(dim=1)
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        return x
'''  
patch_len=16, N=4, d_model=256, h=4, dropout=0.1, d_decode=64
'''
if __name__ == '__main__':
    input1024 = torch.randn((1,2,1024))
    # input128 = torch.randn((1, 2, 128))
    # net = transDownstream(patch_len=4,h=4,d_model=256,N=4,dropout=0.1)
    # out = net(input128)
    # print(out.shape)
    encoder = transEncoder(patch_len=4,h=4,d_model=256,N=4,dropout=0.1)
    mid = encoder(input1024)
    print(mid.shape)

    # decoder = transDecoder(h=4,d_model=256,d_decode=64,dropout=0.1,N=2)
    # output = decoder(mid)
    # print(output.shape)