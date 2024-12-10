import numpy as np
import scipy.sparse as sp
import os

# 这段代码实现了一个推荐系统数据集的加载和预处理流程，包括读取用户-商品交互文件、过滤交互记录、划分训练/验证/测试集以及构建邻接矩阵

def to_np(x):
    """
    将PyTorch张量转换为NumPy数组。

    参数:
        x (torch.Tensor): 要转换的PyTorch张量。

    返回:
        numpy.ndarray: 转换后的NumPy数组。
    """
    return x.data.cpu().numpy()


def dict_set(base_dict, user_id, item_id, val):
    """
    向字典中添加或更新键值对。

    参数:
        base_dict (dict): 目标字典。
        user_id (int): 用户ID。
        item_id (int): 商品ID。
        val (any): 存储的值。
    """
    if user_id in base_dict:
        base_dict[user_id][item_id] = val
    else:
        base_dict[user_id] = {item_id: val}


def list_to_dict(base_list):
    """
    将列表转换为嵌套字典，其中每个元素是一个三元组 (user_id, item_id, value)。

    参数:
        base_list (list of tuple): 包含(user_id, item_id, value)的列表。

    返回:
        dict: 嵌套字典，表示用户的商品交互信息。
    """
    result = {}
    for user_id, item_id, value in base_list:
        dict_set(result, user_id, item_id, value)
    return result


def read_interaction_file(f):
    """
    从文件中读取用户-商品交互信息，并将其转换为三元组列表。

    参数:
        f (file object): 文件对象，包含每行一个用户的交互信息。

    返回:
        list of tuple: 每个元素是(user_id, item_id, rating)形式的三元组。
    """
    total_interactions = []
    for user_id, line in enumerate(f.readlines()):
        items = line.strip().split(' ')[1:]  # 去除换行符并分割字符串，忽略第一个元素（假设是用户ID）
        for item in items:
            item_id = item
            total_interactions.append((user_id, item_id, 1))  # 隐式反馈，所有交互都标记为1
    return total_interactions


def get_count_dict(total_interactions):
    """
    统计每个用户和商品的交互次数。

    参数:
        total_interactions (list of tuple): 用户-商品交互三元组列表。

    返回:
        tuple: 两个字典，分别记录每个用户和商品的交互次数。
    """
    user_count_dict, item_count_dict = {}, {}

    for interaction in total_interactions:
        user, item, _ = interaction

        if user not in user_count_dict:
            user_count_dict[user] = 0
        if item not in item_count_dict:
            item_count_dict[item] = 0

        user_count_dict[user] += 1
        item_count_dict[item] += 1

    return user_count_dict, item_count_dict


def filter_interactions(total_interaction_tmp, user_count_dict, item_count_dict, min_count=[5, 0]):
    """
    根据最小交互次数过滤用户和商品，并重新编号。

    参数:
        total_interaction_tmp (list of tuple): 未过滤的用户-商品交互三元组列表。
        user_count_dict (dict): 用户交互次数统计字典。
        item_count_dict (dict): 商品交互次数统计字典。
        min_count (list of int): 最小用户和商品交互次数阈值。

    返回:
        tuple: 过滤后用户数、商品数、用户到新ID映射、商品到新ID映射及过滤后的交互三元组列表。
    """
    total_interactions = []
    user_to_id, item_to_id = {}, {}
    user_count, item_count = 0, 0

    for line in total_interaction_tmp:
        user, item, rating = line

        # 过滤掉交互次数不足的用户和商品
        if user_count_dict[user] < min_count[0]:
            continue
        if item_count_dict[item] < min_count[1]:
            continue

        if user not in user_to_id:
            user_to_id[user] = user_count
            user_count += 1

        if item not in item_to_id:
            item_to_id[item] = item_count
            item_count += 1

        user_id = user_to_id[user]
        item_id = item_to_id[item]
        rating = 1.

        total_interactions.append((user_id, item_id, rating))

    return user_count, item_count, user_to_id, item_to_id, total_interactions


def load_dataset(path, filename, train_ratio=0.5, min_count=[0, 0], random_seed=0):
    """
    加载数据集，进行数据预处理，包括读取文件、过滤、划分训练/验证/测试集等。

    参数:
        path (str): 数据文件路径。
        filename (str): 数据文件名。
        train_ratio (float): 训练集占比，默认0.5。
        min_count (list of int): 最小用户和商品交互次数阈值。
        random_seed (int): 随机种子，用于保证结果可复现。

    返回:
        tuple: 用户数、商品数、训练集、验证集、测试集。
    """
    np.random.seed(random_seed)
    test_ratio = (1. - train_ratio) / 2  # 测试集和验证集各占剩余部分的一半

    with open(os.path.join(path, filename), 'r') as f:
        total_interaction_tmp = read_interaction_file(f)

    user_count_dict, item_count_dict = get_count_dict(total_interaction_tmp)
    user_count, item_count, user_to_id, item_to_id, total_interactions = filter_interactions(total_interaction_tmp,
                                                                                             user_count_dict,
                                                                                             item_count_dict,
                                                                                             min_count=min_count)

    total_mat = list_to_dict(total_interactions)
    train_mat, valid_mat, test_mat = {}, {}, {}

    for user in total_mat:
        items = list(total_mat[user].keys())
        np.random.shuffle(items)

        num_test_items = int(len(items) * test_ratio)
        test_items = items[:num_test_items]
        valid_items = items[num_test_items: num_test_items * 2]
        train_items = items[num_test_items * 2:]

        for item in test_items:
            dict_set(test_mat, user, item, 1)

        for item in valid_items:
            dict_set(valid_mat, user, item, 1)

        for item in train_items:
            dict_set(train_mat, user, item, 1)

    train_mat_t = {}

    for user in train_mat:
        for item in train_mat[user]:
            dict_set(train_mat_t, item, user, 1)

    for user in list(valid_mat.keys()):
        for item in list(valid_mat[user].keys()):
            if item not in train_mat_t:
                del valid_mat[user][item]
        if len(valid_mat[user]) == 0:
            del valid_mat[user]
            del test_mat[user]

    for user in list(test_mat.keys()):
        for item in list(test_mat[user].keys()):
            if item not in train_mat_t:
                del test_mat[user][item]
        if len(test_mat[user]) == 0:
            del test_mat[user]
            del valid_mat[user]

    return user_count, item_count, train_mat, valid_mat, test_mat


def build_adjmat(user_count, item_count, train_mat, selfloop_flag=True):
    """
    构建用户-商品交互的邻接矩阵，并对其进行归一化处理。

    参数:
        user_count (int): 用户数量。
        item_count (int): 商品数量。
        train_mat (dict): 训练集的用户-商品交互字典。
        selfloop_flag (bool): 是否在图中加入自环。

    返回:
        scipy.sparse.csr_matrix: 归一化后的用户-商品交互邻接矩阵。
    """
    R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
    for user in train_mat:
        for item in train_mat[user]:
            R[user, item] = 1
    R = R.tolil()

    adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    adj_mat[:user_count, user_count:] = R
    adj_mat[user_count:, :user_count] = R.T
    adj_mat = adj_mat.todok()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    if selfloop_flag:
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    else:
        norm_adj_mat = normalized_adj_single(adj_mat)

    return norm_adj_mat.tocsr()