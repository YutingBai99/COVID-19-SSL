from sklearn.metrics import roc_auc_score

def misc(vote_pred,targetlist,vote_score):
    TP = ((vote_pred == 1) & (targetlist == 1)).sum()
    TN = ((vote_pred == 0) & (targetlist == 0)).sum()
    FN = ((vote_pred == 0) & (targetlist == 1)).sum()
    FP = ((vote_pred == 1) & (targetlist == 0)).sum()

    #print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
    #print('TP+FP', TP + FP)
    #print('precision', p)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    #print('recall', r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    #print('F1', F1)
    #print('acc', acc)
    AUC = roc_auc_score(targetlist, vote_score)
    #print('AUCp', roc_auc_score(targetlist, vote_pred))
    #print('AUC', AUC)
    specifificity = TN/(FP+TN)
    return r, p, F1, acc, AUC, specifificity