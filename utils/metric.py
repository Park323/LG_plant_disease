from torch import argmax, long, cat
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score
from torch.nn.modules.loss import MultiLabelSoftMarginLoss

def accuracy_function(real, pred, preprocess=None, inference=False, *args):    
    pred = argmax(pred, dim=1).cpu()
    if inference: return [preprocess.label_decoder(p) for p in pred]
    real = real.cpu()
    score = f1_score(real, pred, average='macro')
    return score

# def sep_metric(real, pred, preprocess, inference=False):
    p_crop, p_disease, p_risk = pred[:,:6], pred[:,6:27], pred[:,27:31]
    p_crop = argmax(p_crop, dim=1)
    p_disease = argmax(p_disease, dim=1)
    p_risk = argmax(p_risk, dim=1)
    pred = cat((p_crop, p_disease, p_risk), dim=1).cpu().numpy()
    pred = [preprocess.label_decoder(x) for x in pred]
    if inference: return pred
    pred = [preprocess.label_dict[x] for x in pred]
    
    r_crop, r_disease, r_risk = real[:,:6], real[:,6:27], real[:,27:31]
    r_crop = argmax(r_crop, dim=1)
    r_disease = argmax(r_disease, dim=1)
    r_risk = argmax(r_risk, dim=1)
    real = cat((p_crop, p_disease, p_risk), dim=1).cpu().numpy()
    real = [preprocess.label_decoder(x) for x in real]
    real = [preprocess.label_dict[x] for x in real]
    
    score = f1_score(real, pred, average='macro')
    return score

def ce_loss(outputs, labels):
    CEL = CrossEntropyLoss()
    return CEL(outputs, labels)

# def seperated_loss(outputs, labels, l1=0.3, l2=0.3, l3=0.3):
    # outputs = [(crop set), (risk set), (disease set)]
    # labels  = [(crop set), (disease set), (risk set)]
    CEL = CrossEntropyLoss()
    MSE = MSELoss()
    loss = l1 * CEL(outputs[0], labels[0]) + l2 * MSE(outputs[2], labels[1]) + l3 * CEL(outputs[1], labels[2])
    
    return loss

def multi_label_loss(outputs, labels):
    MSL = MultiLabelSoftMarginLoss()
    loss = MSL(outputs, labels)
    return loss

# def lab_dr_loss(outputs, labels):
#     dr_pred = cat((outputs[:,6:27],outputs[:,27:31]), dim=1)
#     dr_label= cat((labels[:,6:27],labels[:,27:31]), dim=1)
#     return multi_label_loss(dr_pred, dr_label)

# def lab_cat_loss(outputs, labels):
#     return multi_label_loss(outputs, cat((labels[:,:6],labels[:,31:]), dim=1))