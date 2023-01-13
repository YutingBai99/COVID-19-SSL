import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import shutil


def train(optimizer, epoch, modelname, model, train_loader, device, mixup):
    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, (data, target, indexs) in enumerate(train_loader):

        # move data to device
        data = data.to(device)
        target = target.to(device)

        #mixup
        if mixup:
            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(data.size(0))
            images_a, images_b = data, data[index]
            labels_a, labels_b = target, target[index]
            data = lam * images_a + (1 - lam) * images_b


        optimizer.zero_grad()
        output = model(data)

        criteria = nn.CrossEntropyLoss()
        if mixup:
            loss = lam * criteria(output, labels_a) + (1 - lam) * criteria(output, labels_b)
        else:
            loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

    print('Train set: epoch: {}, Average loss: {:.10f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))


def val(model, val_loader, device):
    model.eval()
    test_loss = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, (data, target, indexs) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist


def test(model, test_loader, device):
    model.eval()
    test_loss = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, (data, target, _) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)
    #真正的标签，两个类的概率，预测的类
    return targetlist, scorelist, predlist

def save_checkpoint(state, is_best, checkpoint, itr):
    filename=f'checkpoint_{itr}.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,f'model_best_{itr}.pth.tar'))