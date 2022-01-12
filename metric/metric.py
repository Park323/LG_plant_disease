from torch import argmax
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import f1_score

def accuracy_function(real, pred):    
    real = real.cpu()
    pred = argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

def jk_loss(outputs, labels, l1=0.3, l2=0.3, l3=0.3):
    # outputs = [(crop set), (risk set), (disease set)]
    # labels  = [(crop set), (disease set), (risk set)]
    CEL = CrossEntropyLoss()
    MSE = MSELoss()
    loss = l1 * CEL(outputs[0], labels[0]) + l2 * MSE(outputs[2], labels[1]) + l3 * CEL(outputs[1], labels[2])
    
    return loss