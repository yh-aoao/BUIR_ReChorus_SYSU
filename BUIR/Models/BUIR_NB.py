import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码实现了一个基于图卷积网络（Graph Convolutional Network, GCN）的双向用户-商品表示模型（BUIR_NB）
# 它使用了自监督学习和对比学习的思想来增强用户和商品的表示。
# 与之前的 BUIR_ID 模型不同，BUIR_NB 使用了图神经网络（Graph Neural Network, GNN）来建模用户-商品交互图，并通过多层传播来获取更丰富的节点表示。

class BUIR_NB(nn.Module):
    def __init__(self, user_count, item_count, latent_size, norm_adj, momentum, n_layers=3, drop_flag=False):
        """
        初始化BUIR_NB模型。

        参数:
            user_count (int): 用户总数。
            item_count (int): 商品总数。
            latent_size (int): 隐向量维度大小。
            norm_adj (scipy.sparse.csr_matrix): 归一化后的用户-商品交互邻接矩阵。
            momentum (float): 动量参数，用于更新目标网络。
            n_layers (int): 图卷积层数，默认为3。
            drop_flag (bool): 是否启用dropout，默认为False。
        """
        super(BUIR_NB, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.momentum = momentum

        # 定义在线和目标编码器，都是LGCN_Encoder实例
        self.online_encoder = LGCN_Encoder(user_count, item_count, latent_size, norm_adj, n_layers, drop_flag)
        self.target_encoder = LGCN_Encoder(user_count, item_count, latent_size, norm_adj, n_layers, drop_flag)

        # 定义预测器，用于转换在线编码器的输出
        self.predictor = nn.Linear(latent_size, latent_size)

        self._init_target()  # 初始化目标编码器参数

    def _init_target(self):
        """
        将在线编码器的参数复制到目标编码器，并冻结目标编码器的参数。
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)  # 复制在线编码器参数到目标编码器
            param_t.requires_grad = False  # 冻结目标编码器参数

    def _update_target(self):
        """
        使用动量更新规则更新目标编码器参数。
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)  # 更新目标编码器参数

    def forward(self, inputs):
        """
        前向传播函数，计算用户和商品的在线和目标表示。

        参数:
            inputs (dict): 包含'user'和'item'键的字典，分别对应用户ID和商品ID。

        返回:
            tuple: 用户和商品的在线和目标表示。
        """
        u_online, i_online = self.online_encoder(inputs)  # 在线编码器的输出
        u_target, i_target = self.target_encoder(inputs)  # 目标编码器的输出
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target  # 返回经过预测器转换的在线表示和目标表示

    @torch.no_grad()
    def get_embedding(self):
        """
        获取所有用户的在线表示和所有商品的在线表示。

        返回:
            tuple: 用户和商品的在线表示，经过预测器转换后的用户和商品表示。
        """
        u_online, i_online = self.online_encoder.get_embedding()  # 获取在线编码器的嵌入
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online  # 返回经过预测器转换的在线表示

    def get_loss(self, output):
        """
        计算对比损失。

        参数:
            output (tuple): 用户和商品的在线和目标表示。

        返回:
            torch.Tensor: 对比损失。
        """
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)  # 归一化用户在线表示
        u_target = F.normalize(u_target, dim=-1)  # 归一化用户目标表示
        i_online = F.normalize(i_online, dim=-1)  # 归一化商品在线表示
        i_target = F.normalize(i_target, dim=-1)  # 归一化商品目标表示

        # 欧氏距离可以由归一化向量之间的负内积代替
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)  # 用户-商品对之间的对比损失
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)  # 商品-用户对之间的对比损失

        return (loss_ui + loss_iu).mean()  # 返回平均对比损失


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
        # self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
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
            torch.sparse.FloatTensor: 稀疏张量。
        """
        coo = X.tocoo()  # 转换为COO格式
        i = torch.LongTensor([coo.row, coo.col])  # 稀疏矩阵的行和列索引
        v = torch.from_numpy(coo.data).float()  # 稀疏矩阵的值
        return torch.sparse.FloatTensor(i, v, coo.shape)  # 创建稀疏张量

    def sparse_dropout(self, x, rate, noise_shape):
        """
        对稀疏张量进行dropout操作。

        参数:
            x (torch.sparse.FloatTensor): 输入的稀疏张量。
            rate (float): dropout比率。
            noise_shape (tuple): dropout的形状。

        返回:
            torch.sparse.FloatTensor: 经过dropout处理的稀疏张量。
        """
        random_tensor = 1 - rate  # 生成随机张量
        # random_tensor += torch.rand(noise_shape).cuda()
        random_tensor += torch.rand(noise_shape)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)  # 生成dropout掩码
        i = x._indices()  # 稀疏张量的索引
        v = x._values()  # 稀疏张量的值

        # 应用dropout掩码
        i = i[:, dropout_mask]
        v = v[dropout_mask]

        # out = torch.sparse.FloatTensor(i, v, x.shape).cuda()  # 创建新的稀疏张量
        out = torch.sparse.FloatTensor(i, v, x.shape)
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
