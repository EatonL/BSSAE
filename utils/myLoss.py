import torch
import torch.nn.functional as F
import torch.nn as nn

def loss_function(x_low, x_high, output, input, batch_size, criterion1, criterion2):
    logits, labels = info_nce_loss(x_low,x_high,batch_size)
    reconstruct_loss = criterion1(output, input.float())
    contrast_loss = 20 * criterion2(logits, labels)
    loss = torch.add(reconstruct_loss, contrast_loss)
    return loss, reconstruct_loss, contrast_loss

def multiLossTrain(x_low, x_high, output, input):
    log_vars = nn.Parameter(torch.zeros((2)))

    precision1 = torch.exp(-log_vars[0])
    reconstruct_loss = torch.sum(precision1*(x_low-x_high)**2.+log_vars[0], -1)
    precision2 = torch.exp(-log_vars[1])
    contrast_loss = torch.sum(precision2*(output-input.float())**2.+log_vars[1], -1)
    reconstruct_loss = torch.mean(reconstruct_loss)
    contrast_loss = torch.mean(contrast_loss)
    loss = torch.add(reconstruct_loss, contrast_loss)
    return loss, reconstruct_loss, contrast_loss, precision1, precision2

def multiLossVal(x_low, x_high, output, input, precision1, precision2):
    log_vars = nn.Parameter(torch.zeros((2)))

    precision1 = precision1.item()
    reconstruct_loss = torch.sum(precision1*(x_low-x_high)**2.+log_vars[0], -1)
    precision2 = precision2.item()
    contrast_loss = torch.sum(precision2*(output-input.float())**2.+log_vars[1], -1)
    reconstruct_loss = torch.mean(reconstruct_loss)
    contrast_loss = torch.mean(contrast_loss)
    loss = torch.add(reconstruct_loss, contrast_loss)
    return loss, reconstruct_loss, contrast_loss

def info_nce_loss(features1, features2, batch_size, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = torch.cat([features1.view(features1.shape[0],-1),features2.view(features2.shape[0],-1)], dim=0)
    features = F.normalize(features, dim=1)

    # print('shape of features: ',features.shape)

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

    # neg = ((lam*(positives + negatives))**q) / q
    # pos = -(positives**q) / q
    # loss = pos.mean() + neg.mean()

    return logits, labels


