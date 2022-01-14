from torch import argmax, long, cat
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score
from torch.nn.modules.loss import MultiLabelSoftMarginLoss

def accuracy_function(real, pred, preprocess, inference=False, *args):    
    pred = argmax(pred, dim=1).cpu()
    if inference: return [preprocess.label_decoder(p) for p in pred]
    real = real.cpu()
    score = f1_score(real, pred, average='macro')
    return score

def seperated_metric(real, pred, preprocess, inference=False):
    pred = [(argmax(x[0], dim=1), argmax(x[1], dim=1), max(0,min(round(x[2]),3))) for x in pred.cpu().numpy()]
    pred = [preprocess.label_decoder(x) for x in pred]
    pred = [preprocess.label_dict[x] for x in pred]
    
    real = [preprocess.label_decoder(x) for x in real.cpu()]
    real = [preprocess.label_dict[x] for x in real]
    
    score = f1_score(real, pred, average='macro')
    return score

def lab_metric(real, pred, preprocess, inference=False):
    p_crop, p_disease, p_risk = pred[:,:6], pred[:,6:27], pred[:,27:31]
    p_crop = argmax(p_crop, dim=1).tolist()
    p_disease = argmax(p_disease, dim=1).tolist()
    p_risk = argmax(p_risk, dim=1).tolist()
    pred = [(preds[0], preds[1], preds[2]) for preds in list(zip(p_crop, p_disease, p_risk))]
    
    pred = [preprocess.label_decoder(x) for x in pred]
    if inference: return pred
    pred = [preprocess.label_dict[x] for x in pred]
    
    r_crop, r_disease, r_risk = real[:,:6], real[:,6:27], real[:,27:31]
    r_crop = argmax(r_crop, dim=1).tolist()
    r_disease = argmax(r_disease, dim=1).tolist()
    r_risk = argmax(r_risk, dim=1).tolist()
    real = [preprocess.label_decoder(x) for x in list(zip(r_crop, r_disease, r_risk))]
    real = [preprocess.label_dict[x] for x in real]
    
    score = f1_score(real, pred, average='macro')
    return score

def lab_cat_metric(real, pred, preprocess, inference=False):
    p_crop, p_area, p_grow = pred[:,:6], pred[:,6:13], pred[:,13:22]
    p_crop = argmax(p_crop, dim=1).tolist()
    p_area = argmax(p_area, dim=1).tolist()
    
    r_crop, r_area, r_grow = real[:,:6], real[:,6:13], real[:,13:22]
    r_crop = argmax(r_crop, dim=1).tolist()
    r_area = argmax(r_area, dim=1).tolist()
    
    score_c = f1_score(r_crop, p_crop, average='macro')
    score_a = f1_score(r_area, p_area, average='macro')
    
    score = ( score_c + score_a ) / 2
    
    return score

def base_loss(outputs, labels):
    CEL = CrossEntropyLoss()
    return CEL(outputs, labels)

def seperated_loss(outputs, labels, l1=0.3, l2=0.3, l3=0.3):
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

def lab_dr_loss(outputs, labels):
    dr_pred = cat((outputs[:,6:27],outputs[:,27:31]), dim=1)
    dr_label= cat((labels[:,6:27],labels[:,27:31]), dim=1)
    return multi_label_loss(dr_pred, dr_label)

def lab_cat_loss(outputs, labels):
    return multi_label_loss(outputs, cat((labels[:,:6],labels[:,31:]), dim=1))