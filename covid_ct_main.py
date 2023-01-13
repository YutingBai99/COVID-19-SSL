import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle
from efficientnet_pytorch import EfficientNet
from train_util import train,val,test, save_checkpoint
from misc import misc
from data_process import CovidCTDataset, CovidCTDatasetSSL, split_data, get_data_path, list_split
from pseudo_labeling_util import pseudo_labeling
from datetime import datetime

torch.cuda.empty_cache()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


device = 'cuda'

def run_test(out):
    vote_pred = np.zeros(test_set.__len__())
    vote_score = np.zeros(test_set.__len__())

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    test_total_epoch = 10
    for epoch in range(1, test_total_epoch + 1):

        targetlist, scorelist, predlist = test(model, test_loader, device)
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist

        if epoch % votenum == 0:
            # major vote
            vote_pred[vote_pred <= (votenum / 2)] = 0
            vote_pred[vote_pred > (votenum / 2)] = 1

            r, p, F1, acc, AUC, specifificity = misc(vote_pred, targetlist, vote_score)

            vote_pred = np.zeros((1, test_set.__len__()))
            vote_score = np.zeros(test_set.__len__())

            print(
                'Test:  The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}, specifificity: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC, specifificity))

            f = open('{}/{}.txt'.format(out, modelname), 'a+')
            f.write(
                'Test:  The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}, specifificity: {:.4f}\n'.format(
                    epoch, r, p, F1, acc, AUC, specifificity))
            f.close()


if __name__ == '__main__':
    datasets = "SARS-COV-2"    # COVID-CT  SARS-COV-2 Harvard
    exp_name = "th0.985"
    modelname = 'ResNet50'   #efficientNet-b0 Dense169 ResNet50
    out = 'model_result/' + datasets + "_" + modelname + "_" + exp_name
    resume = 'model_result/' + datasets + "_" + modelname + "_" + exp_name
    #resume = ""
    iterations = 10
    total_epoch = 100
    generate_data = True
    batchsize = 10
    #lr = 0.00001
    lr = 0.0001
    th = 0.985
    train_rate = 0.2  #0.1 0.2 1
    mixup = True

    start_itr = 0
    if resume and os.path.isdir(resume):
        #os.listdir取出所有文件和文件夹
        resume_files = os.listdir(resume)
        #取出pseudo_labeling_iteration后面的标号，即迭代次数
        resume_itrs = [int(item.replace('.pkl', '').split("_")[-1]) for item in resume_files if
                       'pseudo_labeling_iteration' in item]
        #接着目前最大的迭代次数训练。
        if len(resume_itrs) > 0:
            start_itr = max(resume_itrs)
        out = resume
    os.makedirs(out, exist_ok=True)

    root_dir = 'datasets/' + datasets + '/data'
    if generate_data:
        txt_train_COVID, txt_test_COVID = split_data(root_dir, datasets, "COVID")
        txt_train_NonCOVID, txt_test_NonCOVID = split_data(root_dir, datasets, "NonCOVID")
    else:
        txt_train_COVID, txt_test_COVID, _, _ = get_data_path(datasets, "COVID")
        txt_train_NonCOVID, txt_test_NonCOVID, _, _ = get_data_path(datasets, "NonCOVID")

    for iter in range(start_itr, iterations):
        train_set = CovidCTDataset(root_dir=root_dir,
                                   txt_COVID=txt_train_COVID,
                                   txt_NonCOVID=txt_train_NonCOVID,
                                   transform=train_transformer)

        test_set = CovidCTDataset(root_dir=root_dir,
                                 txt_COVID=txt_test_COVID,
                                 txt_NonCOVID=txt_test_NonCOVID,
                                 transform=val_transformer)

        if iter==0:
            indexs = train_set.indexs
            train_label_indexs, train_unlabel_indexs = list_split(indexs, ratio=train_rate, shuffle=True)
            label_unlabel_split = {'train_label_indexs': train_label_indexs, 'train_unlabel_indexs': train_unlabel_indexs}

            with open("datasets/" + datasets + "/label_unlabel_split.txt", "wb") as f:
                pickle.dump(label_unlabel_split, f)

            train_label_set = CovidCTDatasetSSL(train_set, indexs=train_label_indexs, pse_idx=None, pse_label=None, transform=train_transformer)
        else:
            label_unlabel_split = pickle.load(open("datasets/" + datasets + "/label_unlabel_split.txt", 'rb'))
            train_label_indexs = label_unlabel_split["train_label_indexs"]
            train_unlabel_indexs = label_unlabel_split["train_unlabel_indexs"]

            pseudo_lbl_dict = pickle.load(open(f'{out}/pseudo_labeling_iteration_{iter-1}.pkl', 'rb'))
            pseudo_idx=pseudo_lbl_dict["pseudo_idx"]
            pseudo_target=pseudo_lbl_dict["pseudo_target"]

            lbl_index = np.array(train_label_indexs + pseudo_idx)
            pseudo_idx = np.array(pseudo_lbl_dict["pseudo_idx"])
            train_label_set = CovidCTDatasetSSL(train_set, indexs=lbl_index, pse_idx=pseudo_idx, pse_label=pseudo_target, transform=train_transformer)

        train_unlabel_set = CovidCTDatasetSSL(train_set, indexs=train_unlabel_indexs, pse_idx=None, pse_label=None, transform=val_transformer)

        if modelname == 'efficientNet-b0':
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
            #model = EfficientNet.from_name('efficientnet-b0')
        elif modelname == 'Dense169':
            model = models.densenet169(pretrained=True).cuda()
        elif modelname == 'ResNet50':
            model = models.resnet50(pretrained=True).cuda()
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
            #model = EfficientNet.from_name('efficientnet-b0')
            modelname = 'efficientNet-b0'
        model = model.cuda()

        with open('{}/{}.txt'.format(out, modelname), 'a+') as f:
            f.write("out: " + out + "\n")
            f.write("train_set: " + str(train_set.__len__()) + "     train_label_set: " + str(train_label_set.__len__()) + "    train_unlabel_set: " + str(train_unlabel_set.__len__()) + "    test_set: " + str(test_set.__len__()))
        print("out: " + out + "\n")
        print("train_set: " + str(train_set.__len__()) + "     train_label_set: " + str(train_label_set.__len__()) + "    train_unlabel_set: " + str(train_unlabel_set.__len__())+"    test_set: " + str(test_set.__len__()))

        train_loader = DataLoader(train_label_set, batch_size=batchsize, drop_last=False, shuffle=True)
        train_label_loader = DataLoader(train_label_set, batch_size=batchsize, drop_last=False, shuffle=True)
        train_unlabel_loader = DataLoader(train_unlabel_set, batch_size=batchsize, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batchsize, drop_last=False, shuffle=False)


        with open('{}/{}.txt'.format(out, modelname), 'a+') as f:
            f.write('\n----------------------------------- iter: {} -----------------------------------\n'.format(iter))
        print('----------------------------------- iter: {} -----------------------------------'.format(iter))

        # train
        votenum = 10

        import warnings
        warnings.filterwarnings('ignore')

        vote_pred = np.zeros(test_set.__len__())
        vote_score = np.zeros(test_set.__len__())

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        start_epoch = 0
        starttime = datetime.now()
        cp_run_time = 0
        if resume and iter == start_itr and os.path.isdir(resume):
            resume_itrs = [int(item.replace('.pth.tar', '').split("_")[-1]) for item in resume_files if
                           'checkpoint_iteration_' in item]
            if len(resume_itrs) > 0:
                checkpoint_itr = max(resume_itrs)
                resume_model = os.path.join(resume, f'checkpoint_iteration_{checkpoint_itr}.pth.tar')
                if os.path.isfile(resume_model) and checkpoint_itr == iter:
                    checkpoint = torch.load(resume_model)
                    cp_run_time = checkpoint['run_time']
                    best_acc = checkpoint['best_acc']
                    start_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])

        for epoch in range(start_epoch, total_epoch):
            train(optimizer, epoch, modelname, model, train_loader, device, mixup)

            targetlist, scorelist, predlist = test(model, test_loader, device)

            vote_pred = vote_pred + predlist
            vote_score = vote_score + scorelist

            best_acc=0.0
            acc =0.0
            if (epoch+1) % votenum == 0:
                vote_pred[vote_pred <= (votenum / 2)] = 0
                vote_pred[vote_pred > (votenum / 2)] = 1
                vote_score = vote_score / votenum

                r, p, F1, acc, AUC, specifificity = misc(vote_pred,targetlist,vote_score)

                vote_pred = np.zeros(test_set.__len__())
                vote_score = np.zeros(test_set.__len__())
                print('The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}, specifificity: {:.4f}'.format(
                        epoch, r, p, F1, acc, AUC, specifificity))

                with open('{}/{}.txt'.format(out, modelname), 'a+') as f:
                    f.write('The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}, specifificity: {:.4f}\n'.format(
                        epoch, r, p, F1, acc, AUC, specifificity))

            if epoch == 500:
                run_test(out)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            model_to_save = model.module if hasattr(model, "module") else model
            # 每次训练存checkpoint和最好的model
            save_checkpoint({
                'run_time': (datetime.now() - starttime).seconds + cp_run_time,
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'acc': acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, out, f'iteration_{str(iter)}')

        checkpoint = torch.load(f'{out}/checkpoint_iteration_{str(iter)}.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.zero_grad()

        pseudo_labeling_nobl_acc, nobl_select_num, pseudo_label_dict = pseudo_labeling(train_unlabel_loader, model, device, iter, th)

        with open(f'{out}/pseudo_labeling_iteration_{iter}.pkl', "wb") as f:
            pickle.dump(pseudo_label_dict, f)

        with open('{}/{}.txt'.format(out, modelname), 'a+') as f:
            f.write('pseudo_label: iter{}: Acc: {:.4f}, number: {:.4f}, th: {:.4f}\n'.format(iter, pseudo_labeling_nobl_acc, nobl_select_num, th))

        run_test(out)


        endtime = datetime.now()
        iter_time = (endtime - starttime).seconds
        day = iter_time // (24 * 60 * 60)
        hours = (iter_time % (24 * 60 * 60)) // (60 * 60)
        minute = (iter_time % (60 * 60)) // 60
        second = iter_time % 60


        with open('{}/{}.txt'.format(out, modelname), 'a+') as f:
            f.write(f'iter time: {iter_time}, {day}D{hours}:{minute}:{second}\n\n')


