import math
import functools
import numpy as np

# 将数据集划分给多个客户端
def clients_indices(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    # 建立样本类别和索引的映射关系
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))
    # 用 dirichlet 采样建立 non-iid 数据分布
    # 这个刚好会把数据索引集分成 20 个子列表，所以后面并没有对类别进行重新划分
    # 不过似乎也不用重新划分，实验需要满足的是整体数据分布是 long_tail，并不在乎每一个client上的数据分布
    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    # 将样本数据数组进行合并，得到一个包含所有样本索引的列表
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    # 获得各个clinet的样本索引集
    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance(idxs, num_split: int):
    # 计算每个client的样本数，和剩余的样本数量
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    # 如果还有余数可以用，就将 num_per_part + 1 分给子列表
    # 如果没有，就将 num_per_part 分给子列表
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    # 打乱样本索引和类别二元数组的顺序
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    # 这一段没有用啊
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    # 一直到这，都没啥用

    # 把索引类别二元数组分成 num_splits 这么多组
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index


    # 
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            # 创建 _n_workers 个空数组 []
            _idx_batch = [[] for _ in range(_n_workers)]
            # 对于每个类别进行一次循环
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                # 获取 _targets 数组中类别等于 _class 的样本的索引，并将其放入一个数组
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                # 进行采样
                try:
                    # 创建服从 Dirichlet 分布的权重比例，得到一个长度为 _n_workers 的一维的 Dirichlet 分布样本
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    # 平衡采样，判断 是否 该工作线程的子列表长度  < (目标样本 / 工作线程数量)
                    # 如果满足条件，则保留，不满足则设为 0
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    # 进行归一化
                    proportions = proportions / proportions.sum()
                    # 计算样本数组的划分点，并放入一个数组中
                    # cumsum 计算累积和，乘以 len(idx_class) 得到经过归一化后的权重在原始样本中的相对位置（按照累积和进行缩放）
                    # astype(int) 将计算出的浮点数换位整数，[:-1] 去掉最后一个元素，得到划分点的数组
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    # 用划分点将样本索引划分成不同的子集，并与之前得到的索引列表合并
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    # 更新 min_size
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch
