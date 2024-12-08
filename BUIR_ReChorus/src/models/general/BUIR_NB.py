# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BUIR
Reference:
    "Bootstrapping User and Item Representations for One-Class Collaborative Filtering"
    Lee et al., SIGIR'2021.
CMD example:
     python .\src\main.py --model_name BUIR_NB --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from models.BaseModel import GeneralModel


class BUIR_NB(GeneralModel):
    # 定义类属性，指定数据读取器、训练运行器以及额外的日志参数
    reader = 'BaseReader'  # 数据读取器名称
    runner = 'BUIRRunner'  # 训练运行器名称
    extra_log_args = ['emb_size', 'momentum']  # 额外记录的日志参数

    @staticmethod
    def parse_model_args(parser):
        """
        解析命令行参数，设置嵌入向量大小和动量更新参数。
        """
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')  # 嵌入向量的维度
        parser.add_argument('--momentum', type=float, default=0.995,
                            help='Momentum update.')  # 动量更新系数
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of GCN layers.')
        parser.add_argument('--drop_flag', action='store_true',
                            help='Enable dropout in GCN layers.')
        return GeneralModel.parse_model_args(parser)

    @staticmethod
    def init_weights(m):
        """
        初始化权重，对线性层使用Xavier初始化，对嵌入层也使用Xavier初始化。
        """
        if 'Linear' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif 'Embedding' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)

    def __init__(self, args, corpus):
        """
        构造函数，初始化BUIR_NB模型。
        """
        super().__init__(args, corpus)
        self.emb_size = args.emb_size  # 嵌入向量的维度
        self.momentum = args.momentum  # 动量更新系数
        self.n_layers = args.n_layers  # 图卷积层数
        self.drop_flag = args.drop_flag  # 是否启用dropout
        self.norm_adj = self._create_normalized_adj(corpus)  # 归一化后的邻接矩阵
        self._define_params()          # 定义模型参数
        self.apply(self.init_weights)  # 应用权重初始化

        # 初始化目标网络参数与在线网络相同，并且不计算梯度
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _create_normalized_adj(self, corpus):
        """
        创建并归一化用户-商品交互邻接矩阵。
        """
        user_item_interactions = corpus.train_clicked_set
        row = []
        col = []
        data = []

        for user, items in user_item_interactions.items():
            for item in items:
                row.append(user)
                col.append(item + self.user_num)
                data.append(1)

        adj = csr_matrix((data, (row, col)), shape=(self.user_num + self.item_num, self.user_num + self.item_num))
        adj_mat = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj_mat = self._normalize_sparse_matrix(adj_mat)
        return norm_adj_mat

    def _normalize_sparse_matrix(self, mat):
        """
        归一化稀疏矩阵。
        """
        rowsum = np.array(mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = csr_matrix(np.diag(d_inv_sqrt))
        normalized_mat = d_mat_inv_sqrt.dot(mat).dot(d_mat_inv_sqrt)
        return normalized_mat

    def _define_params(self):
        """
        定义在线和目标编码器，以及预测器。
        """
        self.online_encoder = LGCN_Encoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers, self.drop_flag)
        self.target_encoder = LGCN_Encoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers, self.drop_flag)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)  # 预测器，用于映射在线表示到目标空间

    def _update_target(self):
        """
        更新目标网络的参数，使用动量更新规则。
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, feed_dict):
        """
        前向传播函数，计算用户-物品的交互分数。
        """
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']  # 获取用户ID和物品ID

        # 计算用户-物品的交互分数
        u_online, i_online = self.online_encoder({'user': user, 'item': items})
        u_online_pred = self.predictor(u_online)
        i_online_pred = self.predictor(i_online)
        u_online = u_online.unsqueeze(1)  # 将 u_online 从 (batch_size, emb_size) 转为 (batch_size, 1, emb_size)
        prediction = (i_online_pred[:, None, :] * u_online).sum(dim=-1) + \
                     (i_online_pred[:, None, :] * u_online).sum(dim=-1)
        out_dict = {'prediction': prediction}  # 输出字典，包含预测结果

        if feed_dict['phase'] == 'train':
            # 如果是在训练阶段，则还需要计算在线表示和目标表示
            u_target, i_target = self.target_encoder({'user': user, 'item': items})
            out_dict.update({
                'u_online': u_online_pred,
                'u_target': u_target,
                'i_online': i_online_pred,
                'i_target': i_target
            })
        return out_dict

    def loss(self, output):
        """
        计算损失函数，基于在线表示和目标表示之间的负内积。
        """
        u_online, u_target = output['u_online'], output['u_target']
        i_online, i_target = output['i_online'], output['i_target']

        # 对在线和目标表示进行L2归一化
        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        # 使用负内积代替欧几里得距离来计算损失
        loss_ui = 2 - 2 * (u_online * i_target.detach()).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target.detach()).sum(dim=-1)

        # 返回平均损失
        return (loss_ui + loss_iu).mean()

    class Dataset(GeneralModel.Dataset):
        """
        数据集类，不需要采样负样本。
        """
        def actions_before_epoch(self):
            """
            在每个epoch开始前执行的动作，这里为每个用户准备一个空列表来存储负样本。
            """
            self.data['neg_items'] = [[] for _ in range(len(self))]


class LGCN_Encoder(nn.Module):
    """
    LGCN_Encoder 是一个图卷积网络（GCN）编码器，负责在用户-商品交互图上进行多层传播，生成用户和商品的隐向量表示。
    """
    def __init__(self, user_count, item_count, latent_size, norm_adj, n_layers=3, drop_flag=False):
        """
        初始化LGCN_Encoder。

        参数:
            user_count (int): 用户总数。
            item_count (int): 商品总数。
            latent_size (int): 隐向量维度大小。
            norm_adj (scipy.sparse.csr_matrix): 归一化后的用户-商品交互邻接矩阵。
            n_layers (int): 图卷积层数，默认为3。
            drop_flag (bool): 是否启用dropout，默认为False。
        """
        super(LGCN_Encoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.layers = [latent_size] * n_layers  # 每层的隐向量维度

        self.norm_adj = norm_adj  # 归一化后的邻接矩阵
        self.drop_ratio = 0.2  # dropout比率
        self.drop_flag = drop_flag  # 是否启用dropout

        # 初始化用户和商品的嵌入
        self.embedding_dict = self._init_model()

        # 将稀疏邻接矩阵转换为稀疏张量
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

    def _init_model(self):
        """
        初始化用户和商品的嵌入。
        """
        initializer = nn.init.xavier_uniform_  # 使用Xavier初始化
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),  # 用户嵌入
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size))),  # 商品嵌入
        })
        return embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        将稀疏矩阵转换为稀疏张量。

        参数:
            X (scipy.sparse.csr_matrix): 稀疏矩阵。

        返回:
           torch.sparse_coo_tensor: 稀疏张量。
        """
        coo = X.tocoo()  # 转换为COO格式
        indices = np.array([coo.row, coo.col])
        i = torch.LongTensor(indices) # 稀疏矩阵的行和列索引
        v = torch.from_numpy(coo.data).float()  # 稀疏矩阵的值
        return torch.sparse_coo_tensor(i, v, coo.shape)  # 创建稀疏张量

    def sparse_dropout(self, x, rate, noise_shape):
        """
        对稀疏张量进行dropout操作。

        参数:
            x (torch.sparse_coo_tensor): 输入的稀疏张量。
            rate (float): dropout比率。
            noise_shape (tuple): dropout的形状。

        返回:
            torch.sparse_coo_tensor: 经过dropout处理的稀疏张量。
        """
        random_tensor = 1 - rate  # 生成随机张量
        random_tensor += torch.rand(noise_shape)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)  # 生成dropout掩码
        i = x._indices()  # 稀疏张量的索引
        v = x._values()  # 稀疏张量的值

        # 应用dropout掩码
        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape)  # 创建新的稀疏张量
        return out * (1. / (1 - rate))  # 缩放以保持期望值不变

    def forward(self, inputs):
        """
        前向传播函数，计算用户和商品的表示。

        参数:
            inputs (dict): 包含'user'和'item'键的字典，分别对应用户ID和商品ID。

        返回:
            tuple: 用户和商品的表示。
        """
        # 如果启用了dropout，则对邻接矩阵进行dropout操作
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        # 将用户和商品的嵌入拼接在一起
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]  # 存储每一层的嵌入

        # 进行多层图卷积传播
        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)  # 图卷积操作
            all_embeddings += [ego_embeddings]  # 添加当前层的嵌入

        # 将所有层的嵌入堆叠并取平均
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        # 分离用户和商品的嵌入
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        # 获取输入的用户和商品ID对应的嵌入
        users, items = inputs['user'], inputs['item']
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings  # 返回用户和商品的表示

    @torch.no_grad()
    def get_embedding(self):
        """
        获取所有用户的嵌入和所有商品的嵌入。

        返回:
            tuple: 所有用户的嵌入和所有商品的嵌入。
        """
        A_hat = self.sparse_norm_adj  # 不进行dropout

        # 将用户和商品的嵌入拼接在一起
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]  # 存储每一层的嵌入

        # 进行多层图卷积传播
        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)  # 图卷积操作
            all_embeddings += [ego_embeddings]  # 添加当前层的嵌入

        # 将所有层的嵌入堆叠并取平均
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        # 分离用户和商品的嵌入
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings  # 返回所有用户的嵌入和所有商品的嵌入



