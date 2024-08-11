# train.py

import torch
import matplotlib.pyplot as plt
import numpy as np

deviceGpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader, model, loss, optimizer):
    model.train()
    lossSum = 0.0
    for i, batch in enumerate(dataloader):
        batchX = batch[0]
        batchX = batchX.to(deviceGpu)
        
        x = torch.autograd.Variable(batchX, requires_grad=False)
        predSP = model(x)
        
        lossOut = loss(batchX, predSP)       
        lossSum = lossOut.item()+lossSum
        
        optimizer.zero_grad()
        lossOut.backward()
        optimizer.step()
    echoloss = lossSum / len(dataloader)
    
    return echoloss
        
def test_loop(dataloader, model, loss):
    model.eval()
    lossSum = 0.0
    for i, batch in enumerate(dataloader):
        batchX = batch[0]
        batchX = batchX.to(deviceGpu)
        
        x = torch.autograd.Variable(batchX, requires_grad=False)
        predSP = model(x)
        
        lossOut = loss(batchX, predSP)
        lossSum = lossOut.item()+lossSum
    echoloss = lossSum / len(dataloader)
    return echoloss

def plotTraining(histrain, histest, hislr, best, zoomin=0):
    histrain = histrain[zoomin:]
    histest = histest[zoomin:]
    epo = len(histrain)
    index = np.arange(0, epo, 1, dtype='uint')
    maxLoss = max(max(histrain), max(histest))
    minLoss = min(min(histrain), min(histest))
    plt.plot(index, histrain, c='blue', linestyle='solid')
    plt.plot(index, histest, c='red', linestyle='dashed')
    if best in histest:
        plt.scatter(histest.index(best), best, marker='*', c='green')
        plt.text(histest.index(best), best, 'best model', c='green', horizontalalignment='center')
    for i in hislr:
        if i-zoomin > 0:
            i = i - zoomin
            plt.plot([i,i],[minLoss,maxLoss*0.99], c='black', linestyle='solid', linewidth=1.0)
            plt.text(i, maxLoss, i, c='black', horizontalalignment='center')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()