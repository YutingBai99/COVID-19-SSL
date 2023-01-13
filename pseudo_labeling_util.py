import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pseudo_labeling(data_loader, model, device, itr, th):
    pseudo_probs = []
    pseudo_max_probs = []
    pseudo_target = []
    pseudo_idx = []
    true_target = []

    model.eval()

    with torch.no_grad():
        # Predict
        for batch_index, (data, target, indexs) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            out_prob = F.softmax(model(data),dim=1)
            max_prob, out_pred = torch.max(out_prob, dim=1)

            selected_idx = (max_prob >= th)

            pseudo_probs.extend(out_prob[selected_idx].cpu().numpy().tolist())
            pseudo_max_probs.extend(max_prob[selected_idx].cpu().numpy().tolist())
            pseudo_target.extend(out_pred[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx].numpy().tolist())
            true_target.extend(target[selected_idx].cpu().numpy().tolist())

    pseudo_target = np.array(pseudo_target)
    true_target = np.array(true_target)

    nobl_select_num = len(pseudo_idx)
    if nobl_select_num==0:
        pseudo_labeling_nobl_acc=0
    else:
        pseudo_labeling_nobl_acc = (pseudo_target == true_target)*1
        pseudo_labeling_nobl_acc = (sum(pseudo_labeling_nobl_acc)/len(pseudo_labeling_nobl_acc))*100
    print(f'\nPseudo-Labeling (positive),before blance,Acc:{pseudo_labeling_nobl_acc},Total Selected: {nobl_select_num},th: {th}')
    pseudo_label_dict = {'pseudo_idx': pseudo_idx, 'pseudo_target': pseudo_target.tolist()}

    return pseudo_labeling_nobl_acc, nobl_select_num, pseudo_label_dict
