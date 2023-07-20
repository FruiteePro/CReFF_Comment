import numpy as np
from Dataset.dataset import label_indices2indices
import copy


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes
    # print(img_max) . 5000
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            # print('imb_factor:{ib}, classes_idx:{ci}, num:{num}'.format(ib = imb_factor, ci = _classes_idx, num = num))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    # 将训练集的索引类别数组展开成一维数组
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    # 得到每一类别中需要选择的图像数量，根据不平衡因子得到各个类别的样本数量
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)

    # 随机取得每个类别的数据样本
    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    # 将得到的类别索引二维数组展开成一维，并打印总样本数量
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    # 返回每个类别的样本数量和对应的类别索引二维数组
    return img_num_list, list_clients_indices


def get_100_samples(list_label2indices_train, num_classes):
    # 获得每一类图片 100 张的索引
    list_indices = []
    for _class in range(num_classes):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        # idx = indices[:100]
        idx = indices[:5]
        list_indices.append(idx)
    # 将得到的类别索引二维数组展开成一维，并打印总样本数量
    num_list_indices = label_indices2indices(list_indices)
    print('All num_data_train')
    print(len(num_list_indices))
    # 返回每个类别的类别索引二维数组
    return list_indices

def get_imb_samples(list_label2indices_train, num_classes):
    # 获得每一类图片的索引
    list_indices = []
    for _class in range(num_classes):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:int(len(list_label2indices_train[_class]) / 10)]
        list_indices.append(idx)
    # 将得到的类别索引二维数组展开成一维，并打印总样本数量
    num_list_indices = label_indices2indices(list_indices)
    print('All num_data_train')
    print(len(num_list_indices))
    # 返回每个类别的类别索引二维数组
    return list_indices




