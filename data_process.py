import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import random
import numpy as np

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['COVID', 'NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        self.targets = []
        for c in range(self.num_cls):
            cls_list = read_txt(self.txt_path[c])
            cls_path_list = [os.path.join(self.root_dir,self.classes[c],item) for item in cls_list]
            self.img_list.extend(cls_path_list)
            self.targets.extend([c for i in range(len(cls_list))])
        self.transform = transform
        self.indexs = [i for i in range(len(self.img_list))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx], -1

class CovidCTDatasetSSL(Dataset):
    def __init__(self, base_dataset, indexs, pse_idx, pse_label,transform):
        super().__init__()
        self.img_list = np.array(base_dataset.img_list)
        self.targets = np.array(base_dataset.targets)
        self.transform = transform

        if pse_label is not None:
            pse_label = np.array(pse_label)
            self.targets[pse_idx] = pse_label

        if indexs is not None and len(indexs)!=0:
            indexs = np.array(indexs)
            self.img_list = np.array(self.img_list)[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs =  np.arange(len(self.img_list))

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, index):
        img_path, target = self.img_list[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, self.indexs[index]



def list_split(full_list, ratio, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def writefile(file_path, data):
    f = open(file_path, 'w+')
    f.write("\n".join(data))
    f.close()

def get_data_path(datasets, class_name):
    root_split_dir = "datasets/" + datasets + "/Data-split/" + class_name
    txt_train_class = root_split_dir + "/train.txt"
    txt_test_class = root_split_dir + "/test.txt"

    txt_train_label_class = root_split_dir + "/train_label.txt"
    txt_train_unlabel_class = root_split_dir + "/train_unlabel.txt"
    return txt_train_class, txt_test_class, txt_train_label_class, txt_train_unlabel_class

def split_data(root_dir,datasets,class_name):
    filePath = root_dir + '/' + class_name
    data_list = os.listdir(filePath)
    data_list.sort()

    txt_train_class, txt_test_class, txt_train_label_class, txt_train_unlabel_class = get_data_path(datasets, class_name)

    train_data, test_data = list_split(data_list, ratio=0.8, shuffle=False)

    writefile(txt_train_class, train_data)
    writefile(txt_test_class, test_data)

    train_label_data, train_unlabel_data = list_split(train_data, ratio=0.2, shuffle=False)
    writefile(txt_train_label_class, train_label_data)
    writefile(txt_train_unlabel_class, train_unlabel_data)

    return txt_train_class, txt_test_class