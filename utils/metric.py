from torch import argmax
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score

def accuracy_function(real, pred, *args):    
    real = real.cpu()
    pred = argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

def seperated_metric(real, pred, preprocess):
    pred, real = pred, real[0]
    real = real.cpu()
    pred = argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score
    
    pred = [(argmax(x[0], dim=1), argmax(x[1], dim=1), max(0,min(round(x[2]),3))) for x in pred.cpu().numpy()]
    pred = [preprocess.label_decoder(x) for x in pred]
    pred = [preprocess.label_dict[x] for x in pred]
    
    real = [preprocess.label_decoder(x) for x in real.cpu()]
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

def dense_loss(outputs, labels):
    CEL = CrossEntropyLoss()
    MSE = MSELoss()
    
    return CEL(outputs, labels[0])