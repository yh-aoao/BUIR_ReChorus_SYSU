import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码实现了一个基于自监督学习的双向交互推荐系统（Bidirectional User-Item Representation, BUIR）
# 它使用了对比学习的思想来增强用户和商品表示

class BUIR_ID(nn.Module):
    def __init__(self, user_count, item_count, latent_size, momentum):
        """
        初始化BUIR模型。

        参数:
            user_count (int): 用户总数。
            item_count (int): 商品总数。
            latent_size (int): 隐向量维度大小。
            momentum (float): 动量参数，用于更新目标网络。
        """
        super(BUIR_ID, self).__init__()
        self.user_count = user_count  # 用户数量
        self.item_count = item_count  # 商品数量
        self.latent_size = latent_size  # 隐向量维度
        self.momentum = momentum  # 目标网络更新的动量因子

        # 定义在线和目标网络的嵌入层
        # 嵌入层的作用是将用户ID和商品ID转换为固定长度的向量（称为隐向量），这些向量能够捕捉用户的兴趣和商品的特征。
        # 对于每个用户和商品，我们都有两个嵌入层：一个是在线网络("学生")的嵌入层，另一个是目标网络("老师")的嵌入层
        self.user_online = nn.Embedding(self.user_count, latent_size)  # 用户在线嵌入
        self.user_target = nn.Embedding(self.user_count, latent_size)  # 用户目标嵌入
        self.item_online = nn.Embedding(self.item_count, latent_size)  # 商品在线嵌入
        self.item_target = nn.Embedding(self.item_count, latent_size)  # 商品目标嵌入

        # 定义预测器，用于转换在线网络的输出
        # 为了让在线网络的表示更加丰富，我们在其输出上加了一个预测器，它是一个简单的线性变换。
        # 这个预测器的作用是让在线网络的表示稍微“变形”一下，从而增加模型的表达能力。
        self.predictor = nn.Linear(latent_size, latent_size)

        self._init_model()  # 初始化模型参数
        self._init_target()  # 初始化目标网络参数

    def _init_model(self):
        """
        使用Xavier初始化线性层权重，并初始化偏置为0；使用Xavier初始化嵌入层权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)  # 线性层权重初始化
                nn.init.normal_(m.bias.data)  # 线性层偏置初始化

            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)  # 嵌入层权重初始化

    def _init_target(self):
        """
        将在线网络的参数复制到目标网络，并冻结目标网络的参数。
        """
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data.copy_(param_o.data)  # 复制用户在线网络参数到目标网络
            param_t.requires_grad = False  # 冻结目标网络参数

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data.copy_(param_o.data)  # 复制商品在线网络参数到目标网络
            param_t.requires_grad = False  # 冻结目标网络参数

    def _update_target(self):
        """
        使用动量更新规则更新目标网络参数。
        目标网络的参数并不是直接训练的，而是通过一种叫做动量更新的方式逐步更新。这意味着目标网络的参数会缓慢地跟随在线网络的变化，但不会完全同步。
        这样做可以让目标网络保持一定的稳定性，避免过于频繁的变化影响学习效果。
        动量更新机制确保目标网络的变化是平滑的，避免了过快的更新导致学习不稳定。
        """
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)  # 更新用户目标网络参数

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)  # 更新商品目标网络参数

    def forward(self, inputs):
        """
        前向传播函数，计算用户和商品的在线和目标表示。

        参数:
            inputs (dict): 包含'user'和'item'键的字典，分别对应用户ID和商品ID。

        返回:
            tuple: 用户和商品的在线和目标表示。
        """
        user, item = inputs['user'], inputs['item']

        u_online = self.predictor(self.user_online(user))  # 用户在线表示，通过预测器转换
        u_target = self.user_target(user)  # 用户目标表示
        i_online = self.predictor(self.item_online(item))  # 商品在线表示，通过预测器转换
        i_target = self.item_target(item)  # 商品目标表示

        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        """
        获取所有用户的在线表示和所有商品的在线表示。

        返回:
            tuple: 用户和商品的在线表示，经过预测器转换后的用户和商品表示。
        """
        u_online = self.user_online.weight
        i_online = self.item_online.weight
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        """
        计算对比损失。
        模型通过计算用户和商品表示之间的余弦相似度来衡量它们的相似性。余弦相似度的值在-1到1之间，值越大表示两个向量越相似。

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