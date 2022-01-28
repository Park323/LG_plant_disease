import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score
from torch.nn.modules.loss import MultiLabelSoftMarginLoss

def get_correct_per_class(real, pred, num_class):
    ### Accuracy per Class ###
    correct = np.zeros(num_class)
    total   = np.zeros(num_class)
    for i, rl in enumerate(real):
        total[rl] += 1
        if pred[i]==rl:
            correct[rl] +=1
    return correct, total

def accuracy_function(real, pred, num_class=0, preprocess=None, inference=False, *args, **kwargs):    
    pred = torch.argmax(pred, dim=1).cpu()
    if inference: 
        pred = pred.tolist()
        return [preprocess.label_decoder(p) for p in pred]
    real = real.cpu()
    
    score = f1_score(real, pred, average='macro')
    
    correct, total = get_correct_per_class(real, pred, num_class)
    return score, correct, total

def sequence_f1(real, pred, num_class=0, preprocess=None, inference=False, *args, **kwargs):
    '''
    inputs - tensor (BxLxC)
    '''
    decoder = preprocess.partial_decoder
    
    pred = pred[:,1:].cpu()
    if inference:
        return [f"{decoder(c.item()).strip('#')}_{decoder(d.item()).strip('#')}_{decoder(r.item()).strip('#')}" for c, d, r in pred]
    pred = [preprocess.super_encoder(f"{decoder(c.item()).strip('#')}_{decoder(d.item()).strip('#')}_{decoder(r.item()).strip('#')}", True) for c, d, r in pred]
        
    real = real[:,1:-1].cpu()
    real = [preprocess.super_encoder(f"{decoder(c.item()).strip('#')}_{decoder(d.item()).strip('#')}_{decoder(r.item()).strip('#')}", True) for c, d, r in real]
    
    score = f1_score(real, pred, average='macro')
    
    correct, total = get_correct_per_class(real, pred, num_class)
    return score, correct, total

def ce_loss(outputs, labels, **kwargs):
    CEL = LabelSmoothingLoss(smoothing=kwargs.get('smoothing',0), gamma=kwargs.get('gamma',0))
    return CEL(outputs, labels)

def sequence_loss(outputs, labels, **kwargs):
    outputs = outputs[:,:-1].contiguous().view(-1, outputs.shape[-1]) # (B*(L-1) x C)
    labels = labels[:,1:-1].contiguous().view(-1)
    return ce_loss(outputs, labels, **kwargs)

def multi_label_loss(outputs, labels, **kwagrs):
    MSL = MultiLabelSoftMarginLoss()
    loss = MSL(outputs, labels)
    return loss

def NLLLoss(preds, targets, reduction='mean', weight=None, gamma=0):
    out = torch.zeros_like(targets, dtype=torch.float)
    logs = torch.log(preds)
    for i in range(len(targets)):
        out[i] = torch.float_power(1-preds[i][targets[i]], gamma) * logs[i][targets[i]]
        if weight:
            out[i] = out[i] * weight[targets[i]]
    return -out.mean() if reduction=='mean' else -out.sum()

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, 
                 reduction="mean", weight=None, gamma=0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight
        self.gamma     = gamma

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)
        n = preds.size(-1)
        preds = F.softmax(preds, dim=-1)
        log_preds = torch.log(preds)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = NLLLoss(
            preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
