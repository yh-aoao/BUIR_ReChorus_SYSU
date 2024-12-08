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
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class BUIR(GeneralModel):
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
        构造函数，初始化BUIR模型。
        """
        super().__init__(args, corpus)
        self.emb_size = args.emb_size  # 嵌入向量的维度
        self.momentum = args.momentum  # 动量更新系数
        self._define_params()          # 定义模型参数
        self.apply(self.init_weights)  # 应用权重初始化

        # 初始化目标网络参数与在线网络相同，并且不计算梯度
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _define_params(self):
        """
        定义用户和物品的在线和目标嵌入矩阵，以及预测器和批量归一化层。
        """
        self.user_online = nn.Embedding(self.user_num, self.emb_size)  # 用户在线嵌入
        self.user_target = nn.Embedding(self.user_num, self.emb_size)  # 用户目标嵌入
        self.item_online = nn.Embedding(self.item_num, self.emb_size)  # 物品在线嵌入
        self.item_target = nn.Embedding(self.item_num, self.emb_size)  # 物品目标嵌入
        self.predictor = nn.Linear(self.emb_size, self.emb_size)       # 预测器，用于映射在线表示到目标空间
        self.bn = nn.BatchNorm1d(self.emb_size, eps=0, affine=False, track_running_stats=False)  # 批量归一化层

    def _update_target(self):
        """
        更新目标网络的参数，使用动量更新规则。
        """
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, feed_dict):
        """
        前向传播函数，计算用户-物品的交互分数。
        """
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']  # 获取用户ID和物品ID

        # 计算用户-物品的交互分数
        prediction = (self.predictor(self.item_online(items)) * self.user_online(user)[:, None, :]).sum(dim=-1) + \
                     (self.predictor(self.user_online(user))[:, None, :] * self.item_online(items)).sum(dim=-1)
        out_dict = {'prediction': prediction}  # 输出字典，包含预测结果

        if feed_dict['phase'] == 'train':
            # 如果是在训练阶段，则还需要计算在线表示和目标表示
            u_online = self.user_online(user)
            u_online = self.predictor(u_online)
            u_target = self.user_target(user)
            i_online = self.item_online(items).squeeze(1)
            i_online = self.predictor(i_online)
            i_target = self.item_target(items).squeeze(1)
            out_dict.update({
                'u_online': u_online,
                'u_target': u_target,
                'i_online': i_online,
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