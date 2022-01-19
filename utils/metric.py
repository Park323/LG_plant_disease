import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score
from torch.nn.modules.loss import MultiLabelSoftMarginLoss

def accuracy_function(real, pred, preprocess=None, inference=False, *args, **kwargs):    
    pred = torch.argmax(pred, dim=1).cpu()
    if inference: 
        pred = pred.tolist()
        return [preprocess.label_decoder(p) for p in pred]
    real = real.cpu()
    score = f1_score(real, pred, average='macro')
    return score

def sequence_f1(real, pred, preprocess=None, inference=False, *args, **kwargs):
    '''
    inputs - tensor (BxLxC)
    '''
    decoder = preprocess.partial_decoder
    
    # pred = torch.argmax(pred[:,:-1], dim=2).cpu() #(Bx(L-1))
    pred = pred[:,1:].cpu()
    if inference:
        return [f"{decoder(c.item()).strip('#')}_{decoder(d.item()).strip('#')}_{decoder(r.item()).strip('#')}" for c, d, r in pred]
    pred = [preprocess.super_encoder(f"{decoder(c.item()).strip('#')}_{decoder(d.item()).strip('#')}_{decoder(r.item()).strip('#')}", True) for c, d, r in pred]
        
    real = real[:,1:-1].cpu()
    real = [preprocess.super_encoder(f"{decoder(c.item()).strip('#')}_{decoder(d.item()).strip('#')}_{decoder(r.item()).strip('#')}", True) for c, d, r in real]
    
    score = f1_score(real, pred, average='macro')
    return score

def ce_loss(outputs, labels, **kwargs):
    # CEL = CrossEntropyLoss()
    CEL = LabelSmoothingLoss(smoothing=kwargs.get('smoothing',0))
    return CEL(outputs, labels)

def sequence_loss(outputs, labels, **kwargs):
    outputs = outputs[:,:-1].contiguous().view(-1, outputs.shape[-1]) # (B*(L-1) x C)
    labels = labels[:,1:-1].contiguous().view(-1)
    return ce_loss(outputs, labels, **kwargs)

def multi_label_loss(outputs, labels, **kwagrs):
    MSL = MultiLabelSoftMarginLoss()
    loss = MSL(outputs, labels)
    return loss

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

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
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

# def sep_metric(real, pred, preprocess, inference=False):
#     p_crop, p_disease, p_risk = pred[:,:6], pred[:,6:27], pred[:,27:31]
#     p_crop = torch.argmax(p_crop, dim=1)
#     p_disease = torch.argmax(p_disease, dim=1)
#     p_risk = torch.argmax(p_risk, dim=1)
#     pred = cat((p_crop, p_disease, p_risk), dim=1).cpu().numpy()
#     pred = [preprocess.label_decoder(x) for x in pred]
#     if inference: return pred
#     pred = [preprocess.label_dict[x] for x in pred]
    
#     r_crop, r_disease, r_risk = real[:,:6], real[:,6:27], real[:,27:31]
#     r_crop = torch.argmax(r_crop, dim=1)
#     r_disease = torch.argmax(r_disease, dim=1)
#     r_risk = torch.argmax(r_risk, dim=1)
#     real = cat((p_crop, p_disease, p_risk), dim=1).cpu().numpy()
#     real = [preprocess.label_decoder(x) for x in real]
#     real = [preprocess.label_dict[x] for x in real]
    
#     score = f1_score(real, pred, average='macro')
#     return score

# def seperated_loss(outputs, labels, l1=0.3, l2=0.3, l3=0.3):
#     # outputs = [(crop set), (risk set), (disease set)]
#     # labels  = [(crop set), (disease set), (risk set)]
#     CEL = CrossEntropyLoss()
#     MSE = MSELoss()
#     loss = l1 * CEL(outputs[0], labels[0]) + l2 * MSE(outputs[2], labels[1]) + l3 * CEL(outputs[1], labels[2])
    
#     return loss

# def lab_dr_loss(outputs, labels):
#     dr_pred = cat((outputs[:,6:27],outputs[:,27:31]), dim=1)
#     dr_label= cat((labels[:,6:27],labels[:,27:31]), dim=1)
#     return multi_label_loss(dr_pred, dr_label)

# def lab_cat_loss(outputs, labels):
#     return multi_label_loss(outputs, cat((labels[:,:6],labels[:,31:]), dim=1))