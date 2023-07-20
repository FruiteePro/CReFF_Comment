from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail, get_100_samples, get_imb_samples
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
from Dataset.dataset import label_indices2indices
from Dataset.sample_dirichlet import clients_indices
from Dataset.Gradient_matching_loss import match_loss
import numpy as np
from torch import stack, max, eq, no_grad, Tensor, unsqueeze, split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm
import copy
import torch
import random
import torch.nn as nn
import time
from Dataset.param_aug import DiffAugment
import os


# 定义全局训练模型
class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args,
                 num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature
        # 合成特征，二维张量，大小为 num_classes * num_of_feature, 256，张量类型为浮点数，需要计算梯度，计算设备是device
        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        # 特征的标签，创建了一个全为1的向量，再乘以当前类别索引，得到一个大小为 num_of_feature * num_classes 的向量
        # 用 torch.tensor 将向量转换为张量，不计算梯度，计算设备是device，.view(-1) 将其展开为一维张量
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        # 创建一个特征优化器，用于更新特征的参数
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr_feature)  # optimizer_img for synthetic data
        # 创建了一个交叉熵损失函数对象，指定计算设备为device
        self.criterion = CrossEntropyLoss().to(args.device)
        # 创建了一个基于 ResNet 架构的模型对象
        # resnet_size = 8 表示 ResNet 的大小
        # scaling = 4 表示特征图的缩放因子，控制特征图的通道数
        # save_activations = False 表示是否保存中间激活值，一般在训练过程中不需要保存
        # group_norm_num_groups = None 表示使用 Group Normalization 时的分组数量，如果为 None，则不使用 Group Normalization
        # freeze_bn = False 和 freeze_bn_affine = False 表示是否冻结 Batch Normalization 层的参数，一般在微调或迁移学习时使用
        # num_classes = args.num_classes 表示分类任务中的类别数量
        # 最后将模型指定到 device 上计算
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        # 创建一个优化器，用于更新 re_train 中分类器模型参数
        self.optimizer = SGD(self.syn_model.classifier.parameters(), lr=args.lr_local_training)
        # 创建一个优化器，用于更新 re_train 中模型参数
        self.optimizer_total = SGD(self.syn_model.parameters(), lr=args.lr_local_training)
        # 创建一个线性层对象，作为学习 federated feature 的模型
        # nn.Linear(256, 10) 是一个线性层（全连接层）的构造函数
        self.feature_net = nn.Linear(256, 10).to(args.device)

    # 更新 federated feature
    # 输入参数，全局模型参数 和 client 梯度列表
    def update_feature_syn(self, args, global_params, list_clients_gradient):
        # 获取 feature_net 的当前参数字典
        feature_net_params = self.feature_net.state_dict()
        # 倒序遍历全局模型的参数字典，将分类器偏置和分类器权重取出，放入 feature_net 参数字典中
        for name_param in reversed(global_params):
            if name_param == 'classifier.bias':
                feature_net_params['bias'] = global_params[name_param]
            if name_param == 'classifier.weight':
                feature_net_params['weight'] = global_params[name_param]
                break
        # 将参数字典重新加载进 feature_net 模型
        self.feature_net.load_state_dict(feature_net_params)
        # 将模型设置为训练模式
        self.feature_net.train()
        # 获取 feature_net 的所有参数，并转换成列表
        net_global_parameters = list(self.feature_net.parameters())
        # 创建一个字典，为每个类别创建一个空列表
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        # 遍历 client 梯度列表，将相同类别的梯度聚合到 gw_real_all 字典中
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        # 创建一个字典，为每个勒创建一个空表
        gw_real_avg = {class_index: [] for class_index in range(args.num_classes)}
        # aggregate the real feature gradients
        # 聚合 real feature 的梯度
        # 遍历每个类别
        for i in range(args.num_classes):
            gw_real_temp = []
            # 取出该类别存储梯度的列表
            list_one_class_client_gradient = gw_real_all[i]

            # 如何存在该类别的梯度数据
            if len(list_one_class_client_gradient) != 0:
                # 计算权重，表示每个数据的加权值
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                # 取每个参数的加权平均值
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                # 得到类别 i 的平均梯度
                gw_real_avg[i] = gw_real_temp
        # update the federated features.
        # 更新服务器上的 federated features
        # 迭代多个轮次
        for ep in range(args.match_epoch):
            # 创建损失函数值，将其设为零，并指定设备
            loss_feature = torch.tensor(0.0).to(args.device)
            # 对于每一个类别
            for c in range(args.num_classes):
                # 如果存在该类别的数据
                if len(gw_real_avg[c]) != 0:
                    # 在 feature_net 中获取类别 c 相应的参数，将其重新 reshape 为特征张量
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape((self.num_of_feature, 256))
                    # 创建一个标签张量，将值都设置为 c
                    lab_syn = torch.ones((self.num_of_feature,), device=args.device, dtype=torch.long) * c
                    # 将特征张量输入进特征模型中，得到输出
                    output_syn = self.feature_net(feature_syn)
                    # 用损失函数计算损失
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    # 计算类别 c 的 federated feature 梯度
                    # 计算关于损失 loss_syn 相对于 全局模型参数 的梯度
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    # 计算 federated feature 和 平均梯度 之间的匹配损失，并累加到总损失上
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], args)
            # 将优化器参数梯度清零
            self.optimizer_feature.zero_grad()
            # 对损失函数进行反向传播
            loss_feature.backward()
            # 更新联邦特征模型的参数
            self.optimizer_feature.step()

    # 重新训练模型分类器
    # 接受两个参数，fedavg 平均参数，和本地训练的批次大小
    def feature_re_train(self, fedavg_params, batch_size_local_training):
        # 复制联邦特征模型和标签
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        # 创建 TensorDataset 对象
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        # 创建线性模型 ft_model，输入尺寸为 256，输出为 10
        ft_model = nn.Linear(256, 10).to(args.device)
        # 创建优化器
        optimizer_ft_net = SGD(ft_model.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        # 将模型设置为训练模式
        ft_model.train()
        # 进入训练循环
        for epoch in range(args.crt_epoch):
            # 加载训练数据
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            # 遍历训练数据中的每一个样本
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                # 将数据输入到模型中进行计算，得到输出
                outputs = ft_model(images)
                # 计算输出和标签的损失值
                loss_net = self.criterion(outputs, labels)
                # 将模型梯度清零
                optimizer_ft_net.zero_grad()
                # 将损失值反向传播
                loss_net.backward()
                # 更新模型参数
                optimizer_ft_net.step()
        # 将模型设置为评估模式
        ft_model.eval()
        # 得到模型的参数字典
        feature_net_params = ft_model.state_dict()
        # 倒叙遍历模型，取出分类器的偏置和权重，更新到 fedavg 的参数字典中
        for name_param in reversed(fedavg_params):
            if name_param == 'classifier.bias':
                fedavg_params[name_param] = feature_net_params['bias']
            if name_param == 'classifier.weight':
                fedavg_params[name_param] = feature_net_params['weight']
                break
        # 返回计算得到的 ft_model 和 fedavg 的参数字典
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

    # 初始化全局模型
    # 接受两个参数，本地模型的参数字典 和 对应的本地数据量列表
    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        # 将本地模型的参数复制
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        # 遍历参数列表中的参数
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            # 遍历所有的本地模型参数字典和对应的本地数据量，将参数值乘以本地数据量，存储在列表中
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            # 计算加权平均值，并赋值给参数列表
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        # 返回加权后的模型参数
        return fedavg_global_params

    # 全局模型推理计算
    def global_eval(self, fedavg_params, data_test, batch_size_test):
        # 加载更新后的全局模型
        self.syn_model.load_state_dict(fedavg_params)
        # 将模型设置为评估模式
        self.syn_model.eval()
        # 不进行梯度计算
        with no_grad():
            # 加载数据
            test_loader = DataLoader(data_test, batch_size_test)
            # 统计预测正确的样本数
            num_corrects = 0
            # 遍历数据
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                # 将数据放入模型中预测
                _, outputs = self.syn_model(images)
                # 获取每个样本预测的最大可能的类别
                _, predicts = max(outputs, -1)
                # 统计预测正确的样本数
                # 将预测结果和标签移动到 cpu 上，比较二者是否相等
                # 然后再对列表进行累加，就是相等的样本的个数
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            # 计算准确率
            accuracy = num_corrects / len(data_test)
        # 返回准确率
        return accuracy
    
    # 全局模型推理计算
    def global_eval_no_paramsload(self, data_test, batch_size_test):
        # 将模型设置为评估模式
        self.syn_model.eval()
        # 不进行梯度计算
        with no_grad():
            # 加载数据
            test_loader = DataLoader(data_test, batch_size_test)
            # 统计预测正确的样本数
            num_corrects = 0
            # 遍历数据
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                # 将数据放入模型中预测
                _, outputs = self.syn_model(images)
                # 获取每个样本预测的最大可能的类别
                _, predicts = max(outputs, -1)
                # 统计预测正确的样本数
                # 将预测结果和标签移动到 cpu 上，比较二者是否相等
                # 然后再对列表进行累加，就是相等的样本的个数
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            # 计算准确率
            accuracy = num_corrects / len(data_test)
        # 返回准确率
        return accuracy
    

    # 返回全局模型的状态字典
    def download_params(self):
        return self.syn_model.state_dict()
    
    # 重新训练分类器
    def classifer_re_train(self, batch_size_re_training, epochs_re_training, re_train_data):
        # 定义图像数据的预处理操作，随机剪裁 和 水平翻转
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        # 设置全局模型为评估模式
        self.syn_model.eval()
        # 设置分类器为训练模式
        self.syn_model.classifier.train()

        # 根据轮数进行迭代
        for epoch in range(epochs_re_training):
            data_loader = DataLoader(re_train_data, 
                                     batch_size=batch_size_re_training, 
                                     shuffle=True)
            # 遍历数据
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                # 对图像进行预处理
                images = transform_train(images)
                # 将图像输入到模型中进行计算，得到输出
                _, outputs = self.syn_model(images)
                # 计算输出和标签的损失值
                loss_net = self.criterion(outputs, labels)
                # 将模型梯度清零
                self.optimizer.zero_grad()
                # 将损失值反向传播
                loss_net.backward()
                # 更新模型参数
                self.optimizer.step()

    # 重新训练模型
    def model_re_train(self, batch_size_re_training, epochs_re_training, re_train_data):
        # 定义图像数据的预处理操作，随机剪裁 和 水平翻转
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        # 设置全局模型为训练模式
        self.syn_model.train()

        # 根据轮数进行迭代
        for epoch in range(epochs_re_training):
            data_loader = DataLoader(re_train_data, 
                                     batch_size=batch_size_re_training, 
                                     shuffle=True)
            # 遍历数据
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                # 对图像进行预处理
                images = transform_train(images)
                # 将图像输入到模型中进行计算，得到输出
                _, outputs = self.syn_model(images)
                # 计算输出和标签的损失值
                loss_net = self.criterion(outputs, labels)
                # 将模型梯度清零
                self.optimizer_total.zero_grad()
                # 将损失值反向传播
                loss_net.backward()
                # 更新模型参数
                self.optimizer_total.step()









# 定义本地训练模型
class Local(object):
    # 构造函数，接受两个参数， client 上的数据，和类别分布信息
    def __init__(self,
                 data_client,
                 class_list: int):
        args = args_parser()
        # 存储 client 的数据
        self.data_client = data_client

        # 存储训练指定的设备
        self.device = args.device
        # 存储类别分布信息
        self.class_compose = class_list

        # 创建了一个交叉熵损失函数对象，指定计算设备为device 
        self.criterion = CrossEntropyLoss().to(args.device)

        # 创建了一个基于 ResNet 架构的模型对象
        # resnet_size = 8 表示 ResNet 的大小
        # scaling = 4 表示特征图的缩放因子，控制特征图的通道数
        # save_activations = False 表示是否保存中间激活值，一般在训练过程中不需要保存
        # group_norm_num_groups = None 表示使用 Group Normalization 时的分组数量，如果为 None，则不使用 Group Normalization
        # freeze_bn = False 和 freeze_bn_affine = False 表示是否冻结 Batch Normalization 层的参数，一般在微调或迁移学习时使用
        # num_classes = args.num_classes 表示分类任务中的类别数量
        # 最后将模型指定到 device 上计算 
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(
            args.device)
        # 创建一个特征优化器，用于更新特征的参数
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    # 计算 client 本地的 real feature gradient
    # 传入参数，全局的 params，这里使用的是 re-train model 的参数
    def compute_gradient(self, global_params, args):
        # compute C^k
        # 获得 client 中存在的样本类别和每个类别的样本数量
        list_class, per_class_compose = get_class_num(self.class_compose)  # class组成

        # 建立两个列表，分别存放样本数据和样本标签
        images_all = []
        labels_all = []
        # 用一个类别列表，为每一个存在样本的类别创建一张空列表
        indices_class = {class_index: [] for class_index in list_class}

        # 将图像数据从数据集格式中取出，在最左新加一列作为序号，保存到列表中
        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        # 将图像标签从数据集格式中取出，保存在列表中
        labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]
        # 将图像标签中的数据按照类别分类，将索引存储在 indices_class 中对应的列表中
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        # 将列表中的张量按照维度 dim=0 进行拼接，并将结果存储在 images_all 变量中
        # 假设 images_all 列表的每个张量的形状为 (1, C, H, W)，通过拼接后，最终得到一个形状为 (N, C, H, W) 的张量，其中 N 表示数据的数量
        # 再将张量移动到指定的设备上
        images_all = torch.cat(images_all, dim=0).to(args.device)
        # 将列表转化为张量，指定数据类型为 torch.long，再将张量移动到指定设备上
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        # 从给定的类别中获得随机的 n 张图片
        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        # 将全局参数 global_params 加载到本地模型中
        self.local_model.load_state_dict(global_params)

        # 将本地模型设置为评估模式 (evaluation mode)
        # 模型采用推理模式，向前传播过程中不计算梯度，只输出预测结果
        self.local_model.eval()
        # 将本地模型的分类器设置为训练模式
        self.local_model.classifier.train()
        # 获取本地模型中分类器的参数列表，得到一个迭代器，里面包含了分类器中所有需要进行梯度更新的参数
        net_parameters = list(self.local_model.classifier.parameters())
        # 创建一个交叉熵损失函数
        criterion = CrossEntropyLoss().to(args.device)
        # gradients of all classes
        # 为每个类别新建一个总梯度列表 和一个平均梯度列表
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}

        # choose to repeat 10 times
        # 迭代十次
        for num_compute in range(10):
            # 遍历所有类别和每个类别的样本数，打包成二元组
            for c, num in zip(list_class, per_class_compose):
                # 在类别 c 中随机取出 batch_real 张样本图片
                img_real = get_images(c, args.batch_real)
                # transform
                # 如果需要数据增强
                if args.dsa:
                    # 生成随机种子
                    seed = int(time.time() * 1000) % 100000
                    # 对图片进行数据增强
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                # 建立一个标签向量，其中每个元素都是当前类别 c 的类标签，向量长度等于图片数量
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                # 将 img_real 输入到本地模型中，分别获取特征向量 feature_real 和输出结果 output_real
                feature_real, output_real = self.local_model(img_real)
                # 利用交叉熵损失函数计算输入图像的损失值
                loss_real = criterion(output_real, lab_real)
                # compute the real feature gradients of class c
                # 计算类别 c 的 real feature gradients
                # 使用 pytorch 的自动求导功能，计算损失函数相对于分类器参数的梯度，保存在列表中
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                # 对梯度信息进行处理，将梯度张量与计算图分离，然后对梯度方向进行拷贝，再存放进列表中
                gw_real = list((_.detach().clone() for _ in gw_real))
                # 将得到的梯度信息添加到字典中，记录梯度信息
                truth_gradient_all[c].append(gw_real)
        # 计算每个类别的平均特征梯度
        for i in list_class:
            # 存储每个类别的平均特征梯度
            gw_real_temp = []
            # 获取类别 i 对应的所有特征梯度信息
            gradient_all = truth_gradient_all[i]
            # 计算每个特征梯度的权重
            weight = 1.0 / len(gradient_all)
            # 遍历特征梯度信息列表中的每个特别梯度
            for name_param in range(len(gradient_all[0])):
                # 存储每个样本加权后的特征梯度
                list_values_param = []
                # 遍历每个样本的特征梯度
                for client_one in gradient_all:
                    # 对每个样本的特征梯度进行加权，存储在列表中
                    list_values_param.append(client_one[name_param] * weight)
                # 求和，得到所有样本的加权特征梯度
                value_global_param = sum(list_values_param)
                # 将该类别的平均特征梯度存入列表中
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            # 将第 i 类别打平均特征梯度替换成刚计算得到的值
            truth_gradient_avg[i] = gw_real_temp
        # 返回每个列表的平均特征梯度
        return truth_gradient_avg

    # 本地更新模型
    def local_train(self, args, global_params):
        # 定义图像数据的预处理操作，随机剪裁 和 水平翻转
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        # 将全局模型加载到本地训练模型中
        self.local_model.load_state_dict(global_params)
        # 将本地模型设置为训练模式
        self.local_model.train()
        # 根据设置的本地训练轮数进行迭代
        for _ in range(args.num_epochs_local_training):
            # 使用 pytorch 的 DataLoader 创建一个数据加载器
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            # 遍历每个批次的数据
            for data_batch in data_loader:
                # 将数据拆分成图像和标签
                images, labels = data_batch
                # 指定图像和标签移动到指定设备上
                images, labels = images.to(self.device), labels.to(self.device)
                # 对图像进行预处理
                images = transform_train(images)
                # 将预处理后的图像输入本地模型，获取模型的输出
                _, outputs = self.local_model(images)
                # 使用交叉熵损失函数计算损失
                loss = self.criterion(outputs, labels)
                # 将优化器的梯度缓冲区清零
                self.optimizer.zero_grad()
                # 反向传播，计算梯度
                loss.backward()
                # 根据梯度更新模型参数
                self.optimizer.step()
        # 返回本地模型的参数字典
        return self.local_model.state_dict()


def CReFF():
    args = args_parser()
    # 打印参数
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature}\n '
        'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_net=args.lr_net,
            lr_feature=args.lr_feature,
            num_of_feature=args.num_of_feature,
            match_epoch=args.match_epoch,
            crt_epoch=args.crt_epoch))
    # 创建了一个随机数生成器，用args.seed设置随机数种子
    random_state = np.random.RandomState(args.seed)
    # Load data
    # 数据转换操作，先将输入的图像数据转换为tensor对象，再对每个通道进行归一化
    # 对于RGB三个通道，第一个三元组指定了每个通道的均值，第二个三元组指定了每个通道的标准差
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 下载并加载CIFAR10数据集
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    # Distribute data
    # 获取不同类别的数据分布，得到的是每个类别对应的样本索引
    # list_label2indices是一个二维数组
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    # 得到long_tail数据的类别索引二维数组，类别的数量分布是按照long_tail模式的
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    # 计算各个client的样本索引集
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    # 打印并存储每个设备上数据的类别分布信息
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    # 建立一个全局模型
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          num_of_feature=args.num_of_feature)
    # 建立 clinet 列表
    total_clients = list(range(args.num_clients))
    # 将原始数据存入自定义的数据集中
    # Indices2Dataset 是一个自定义的数据集类，继承自 torch.utils.data.Dataset
    indices2data = Indices2Dataset(data_local_training)
    # 创建一个列表，用于存储训练中的准确率
    re_trained_acc = []
    # 创建了一个全连接模型，输入维度是256，输出维度是10
    # 将这个全连接模型的状态字典保存在 syn_params 变量中
    # 状态字典包含了模型所有的可学习参数（权重，偏置等）以及对应的张量值
    temp_model = nn.Linear(256, 10).to(args.device)
    syn_params = temp_model.state_dict()
    # tqdm 是一个用于显示进度条的 python 库
    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        # 取得全局模型的状态字典
        global_params = global_model.download_params()
        # 将全局模型的状态字典复制一份，存储在 syn_feature_params 变量
        syn_feature_params = copy.deepcopy(global_params)
        # 反向遍历
        # 将 syn_params 的偏置项和权重项赋给全局模型的状态字典
        for name_param in reversed(syn_feature_params):
            if name_param == 'classifier.bias':
                syn_feature_params[name_param] = syn_params['bias']
            if name_param == 'classifier.weight':
                syn_feature_params[name_param] = syn_params['weight']
                break
        # 中 total_clients 列表中随机选择 num_online_clients 个 client 作为在线客户端
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        # 建立三个列表，分别用来
        # 存储 clients 模型梯度；
        # 存储 clients 本地训练参数；
        # 存储 clients 本地数据样本数量；
        list_clients_gradient = []
        list_dicts_local_params = []
        list_nums_local_data = []
        # local training
        # 在 client 上进行本地训练
        for client in online_clients:
            # 根据 client 的索引集加载训练数据
            # 将训练数据存储在 data_client 变量中
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            # 将本地数据样本数量存储在 list_nums_local_data 中
            list_nums_local_data.append(len(data_client))
            # 创建 client 上的本地训练模型
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            # compute the real feature gradients in local data
            # 计算 client 上每个类别的平均特征梯度
            truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), args)
            # 将该 client 的类别平均特征梯度保存到列表中
            list_clients_gradient.append(copy.deepcopy(truth_gradient))
            # local update
            # 本地更新模型，得到本地模型的参数字典
            local_params = local_model.local_train(args, copy.deepcopy(global_params))
            # 将参数字典存入列表中
            list_dicts_local_params.append(copy.deepcopy(local_params))
        # aggregating local models with FedAvg
        # 根据 FedAvg 对本地模型进行聚合
        # 根据设备的数据量对每个设备的模型进行加权聚合
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        # 更新 federated feature 的模型
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient)
        # re-trained classifier
        # 重新训练分类器
        syn_params, ft_params = global_model.feature_re_train(copy.deepcopy(fedavg_params), args.batch_size_local_training)
        # global eval
        # 全局模型推理，计算准确度
        one_re_train_acc = global_model.global_eval(ft_params, data_global_test, args.batch_size_test)
        # 将本轮准确率存储
        re_trained_acc.append(one_re_train_acc)
        # 将计算得到的 fedavg 的参数更新回全局模型中
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        # 每十轮输出一次准确度队列
        if r % 10 == 0:
            print(re_trained_acc)
    print(re_trained_acc)
    # 保存模型
    torch.save(global_model.syn_model.state_dict(), 'global_model_CReFF.pth')
    



def Re_train_Fedavg():
    args = args_parser()
    # 打印参数
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature}\n '
        'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_net=args.lr_net,
            lr_feature=args.lr_feature,
            num_of_feature=args.num_of_feature,
            match_epoch=args.match_epoch,
            crt_epoch=args.crt_epoch))
    # 创建了一个随机数生成器，用args.seed设置随机数种子
    random_state = np.random.RandomState(args.seed)
    # Load data
    # 数据转换操作，先将输入的图像数据转换为tensor对象，再对每个通道进行归一化
    # 对于RGB三个通道，第一个三元组指定了每个通道的均值，第二个三元组指定了每个通道的标准差
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 下载并加载CIFAR10数据集
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    # Distribute data
    # 获取不同类别的数据分布，得到的是每个类别对应的样本索引
    # list_label2indices是一个二维数组
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    # 得到long_tail数据的类别索引二维数组，类别的数量分布是按照long_tail模式的
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                        args.imb_factor, args.imb_type)
    # 计算各个client的样本索引集
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, args.seed)
    # 打印并存储每个设备上数据的类别分布信息
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                                args.num_classes)
    # 获得不平衡数据集
    # list_client2indices = label_indices2indices(list_client2indices)
    # re_train_imb_Data = torch.utils.data.Subset(data_local_training, list_client2indices)
    # print("imbance Data")
    # print(len(re_train_imb_Data))
    # 获得每个类别 100 张图片的数据索引
    list_label2indices_train_new = get_100_samples(copy.deepcopy(list_label2indices), args.num_classes)
    # list_label2indices_train_new = get_imb_samples(copy.deepcopy(list_label2indices_train_new), args.num_classes)
    list_label2indices_train_new = label_indices2indices(list_label2indices_train_new)
    # 获取 re-train 的数据集
    re_train_Data = torch.utils.data.Subset(data_local_training, list_label2indices_train_new)
    # re_train_indices2data = Indices2Dataset(data_local_training)
    # re_train_indices2data.load(list_label2indices_train_new)
    # re_train_Data = re_train_indices2data

    # 建立一个全局模型
    global_model = Global(num_classes=args.num_classes,
                            device=args.device,
                            args=args,
                            num_of_feature=args.num_of_feature)
    # 建立 clinet 列表
    total_clients = list(range(args.num_clients))
    # 将原始数据存入自定义的数据集中
    # Indices2Dataset 是一个自定义的数据集类，继承自 torch.utils.data.Dataset
    indices2data = Indices2Dataset(data_local_training)
    # 创建一个列表，用于存储训练中的准确率
    fedavg_trained_acc = []
    re_train_acc = []
    # tqdm 是一个用于显示进度条的 python 库
    # for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
    #     # 取得全局模型的状态字典
    #     global_params = global_model.download_params()
    #     # 中 total_clients 列表中随机选择 num_online_clients 个 client 作为在线客户端
    #     online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
    #     # 建立三个列表，分别用来
    #     # 存储 clients 本地训练参数；
    #     # 存储 clients 本地数据样本数量；
    #     list_dicts_local_params = []
    #     list_nums_local_data = []
    #     # local training
    #     # 在 client 上进行本地训练
    #     for client in online_clients:
    #         # 根据 client 的索引集加载训练数据
    #         # 将训练数据存储在 data_client 变量中
    #         indices2data.load(list_client2indices[client])
    #         data_client = indices2data
    #         # 将本地数据样本数量存储在 list_nums_local_data 中
    #         list_nums_local_data.append(len(data_client))
    #         # 创建 client 上的本地训练模型
    #         local_model = Local(data_client=data_client,
    #                             class_list=original_dict_per_client[client])
    #         # local update
    #         # 本地更新模型，得到本地模型的参数字典
    #         local_params = local_model.local_train(args, copy.deepcopy(global_params))
    #         # 将参数字典存入列表中
    #         list_dicts_local_params.append(copy.deepcopy(local_params))
    #     # aggregating local models with FedAvg
    #     # 根据 FedAvg 对本地模型进行聚合
    #     # 根据设备的数据量对每个设备的模型进行加权聚合
    #     fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
    #     # global eval
    #     # 全局模型推理，计算准确度
    #     one_re_train_acc = global_model.global_eval(fedavg_params, data_global_test, args.batch_size_test)
    #     # 将本轮准确率存储
    #     fedavg_trained_acc.append(one_re_train_acc)
    #     # 每十轮输出一次准确度队列
    #     if r % 10 == 0:
    #         print(fedavg_trained_acc)

    # # 保存训练模型参数
    # torch.save(global_model.syn_model.state_dict(), 'global_model_fedavg_200_epochs.pth')

    # print(fedavg_trained_acc)
    # 重新训练分类器

    

    path_to_pretrained_file = 'syn_model.pth'
    global_model.syn_model.load_state_dict(torch.load(path_to_pretrained_file))

    one_re_train_acc = global_model.global_eval_no_paramsload(data_global_test, args.batch_size_test)
    print(one_re_train_acc)


    for i in tqdm(range(20), desc='classifer-re-training'):
        # 重新训练分类器
        # global_model.classifer_re_train(100, 1, re_train_Data)

        global_model.classifer_re_train(50, 10, re_train_Data)
        # global_model.model_re_train(100, 10, re_train_Data)
        # global eval
        one_re_train_acc = global_model.global_eval_no_paramsload(data_global_test, args.batch_size_test)
        # 将本轮准确率存储
        re_train_acc.append(one_re_train_acc)
        print(re_train_acc)
    print(re_train_acc)
    # 保存模型
    torch.save(global_model.syn_model.state_dict(), 'global_model_re_train_20_epochs.pth')

        




if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    if args.model == 'CReFF':
        print("CReFF Running")
        CReFF()
    elif args.model == 'Re_train_Fedavg':
        print("Re_train_Fedavg Running")
        Re_train_Fedavg()


