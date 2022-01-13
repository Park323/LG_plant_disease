from torch import argmax, long, cat
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score

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
    p_crop, p_disease, p_risk, _, _ = pred
    p_crop = argmax(p_crop, dim=1).tolist()
    p_disease = argmax(p_disease, dim=1).tolist()
    p_risk = [round(x.item()) for x in p_risk]
    pred = [(preds[0], preds[1], max(0,min(preds[2],3))) for preds in list(zip(p_crop, p_disease, p_risk))]
    pred = [preprocess.label_decoder(x) for x in pred]
    if inference: return pred
    pred = [preprocess.label_dict[x] for x in pred]
    
    r_crop, r_disease, r_risk, _, _ = real
    r_crop = r_crop.tolist()
    r_disease = r_disease.tolist()
    r_risk = r_risk.tolist()
    real = [preprocess.label_decoder(x) for x in list(zip(r_crop, r_disease, r_risk))]
    real = [preprocess.label_dict[x] for x in real]
    
    score = f1_score(real, pred, average='macro')
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

def lab_loss(outputs, labels):
    # CEL = CrossEntropyLoss()
    # MSE = MSELoss()
    wc, wd, wr, wa, wg = 0.25, 0.25, 0.3, 0.1, 0.1    
    loss_c = CrossEntropyLoss()(outputs[0],labels[0])
    loss_d = CrossEntropyLoss()(outputs[1],labels[1])
    loss_r = MSELoss()(outputs[2],labels[2]).to(long)
    loss_a = CrossEntropyLoss()(outputs[3],labels[3])
    loss_g = CrossEntropyLoss()(outputs[4],labels[4])
    
    # print(f"Loss/ Crop:{loss_c:.4f} Disease:{loss_d:.4f} Risk:{loss_r:.4f} Area:{loss_a:.4f} Grow:{loss_g:.4f}")
    
    loss = wc * loss_c + wd * loss_d + wr * loss_r + wa * loss_a + wg * loss_g
    
    return loss