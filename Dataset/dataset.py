import numpy as np
from torch.utils.data.dataset import Dataset
import copy


def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1




def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel

# 将二维的类别样本数组展开成一维
def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

# 继承 torch.utils.data.Dataset 类
class Indices2Dataset(Dataset):
    # 重写构造函数
    # 接受一个参数 dataset
    # 将传入原始数据集保存在 self.dataset 中
    # self.indices 将用来保存要加载的样本的索引
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    # 存储要加载的样本的索引列表
    def load(self, indices: list):
        self.indices = indices

    # 重写了 __getitem__ 方法，该方法在使用索引访问数据集时被调用
    # 根据传入的索引获得实际索引
    # 再从数据集中获得索引对应的图像和标签
    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    # 重写 __len__ 方法，返回加载的样本索引的长度
    def __len__(self):
        return len(self.indices)


# 继承自 torch 的 Dataset
class TensorDataset(Dataset):
    # 构造函数，输入样本图像和标签
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


# 获取各个类别的数据数量
def get_class_num(class_list):
    index = []
    compose = []
    # 遍历 class 列表中的所有元素，如果数量为0就跳过，不为0就添加到列表中
    # 最后返回两个列表，一个是存在的类别列表，另一个是每一种类别的样本数量
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose
