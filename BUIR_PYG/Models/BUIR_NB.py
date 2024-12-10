import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from scipy.sparse import csr_matrix

class BUIR_NB(nn.Module):
    def __init__(self, user_count, item_count, latent_size, norm_adj, momentum, n_layers=3, drop_flag=False):
        super(BUIR_NB, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.momentum = momentum

        # 将norm_adj转换为适合PyTorch Geometric的格式
        coo = norm_adj.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        edge_index = torch.LongTensor(indices)

        # 定义在线和目标编码器
        self.online_encoder = LGCN_Encoder(user_count, item_count, latent_size, edge_index, n_layers, drop_flag)
        self.target_encoder = LGCN_Encoder(user_count, item_count, latent_size, edge_index, n_layers, drop_flag)

        # 定义预测器
        self.predictor = nn.Linear(latent_size, latent_size)

        self._init_target()

    # 其余方法保持不变...
    
    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False 
    
    def _update_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, inputs):
        u_online, i_online = self.online_encoder(inputs)
        u_target, i_target = self.target_encoder(inputs) 
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target 

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)
        
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()
    



class LGCN_Encoder(nn.Module):
    def __init__(self, user_count, item_count, latent_size, edge_index, n_layers=3, drop_flag=False):
        super(LGCN_Encoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.drop_flag = drop_flag

        # 初始化用户和商品的嵌入
        self.embedding_dict = nn.Embedding(user_count + item_count, latent_size)
        nn.init.xavier_uniform_(self.embedding_dict.weight)

        # 创建图卷积层
        self.convs = nn.ModuleList([GCNConv(latent_size, latent_size) for _ in range(n_layers)])

        # 将稀疏邻接矩阵转换为PyTorch Geometric的SparseTensor
        self.edge_index = edge_index  # 这里假设edge_index是已经准备好的边索引

    def forward(self, inputs):
        users, items = inputs['user'], inputs['item']
        x = self.embedding_dict.weight  # 获取所有节点的嵌入

        all_embeddings = [x]
        for conv in self.convs:
            if self.drop_flag:
                x = F.dropout(x, p=0.2, training=self.training)
            x = conv(x, self.edge_index)
            all_embeddings.append(x)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        x = self.embedding_dict.weight  # 获取所有节点的嵌入

        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, self.edge_index)
            all_embeddings.append(x)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings
