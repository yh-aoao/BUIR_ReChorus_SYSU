import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from scipy.sparse import csr_matrix


class BUIR_NB(nn.Module):
    def __init__(self, user_count, item_count, latent_size, norm_adj, momentum, n_layers=3, drop_flag=False):
        super(BUIR_NB, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.momentum = momentum

        # Convert adjacency matrix to edge index
        edge_index = self._convert_adj_to_edge_index(norm_adj)

        # Define online and target encoders, both instances of GAT_Encoder
        self.online_encoder = GAT_Encoder(user_count, item_count, latent_size, edge_index, n_layers, drop_flag)
        self.target_encoder = GAT_Encoder(user_count, item_count, latent_size, edge_index, n_layers, drop_flag)

        # Define predictor to transform the output of the online encoder
        self.predictor = nn.Linear(latent_size, latent_size)

        self._init_target()  # Initialize target encoder parameters

    def _convert_adj_to_edge_index(self, adj_matrix):
        """
        Convert a sparse adjacency matrix to an edge index tensor.

        Parameters:
            adj_matrix (scipy.sparse.csr_matrix): The adjacency matrix.

        Returns:
            torch.LongTensor: The edge index tensor.
        """
        coo = adj_matrix.tocoo()
        row_indices = torch.from_numpy(coo.row.astype(np.int64))
        col_indices = torch.from_numpy(coo.col.astype(np.int64))
        edge_index = torch.stack([row_indices, col_indices], dim=0)
        return edge_index

    def _init_target(self):
        """
        Copy parameters from the online encoder to the target encoder and freeze the target encoder's parameters.
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)  # Copy online encoder parameters to target encoder
            param_t.requires_grad = False  # Freeze target encoder parameters

    def _update_target(self):
        """
        Update the target encoder parameters using the momentum update rule.
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (
                        1. - self.momentum)  # Update target encoder parameters

    def forward(self, inputs):
        """
        Forward pass function, compute the online and target representations of users and items.

        Parameters:
            inputs (dict): A dictionary containing 'user' and 'item' keys corresponding to user IDs and item IDs.

        Returns:
            tuple: Online and target representations of users and items.
        """
        u_online, i_online = self.online_encoder(inputs)  # Output from the online encoder
        u_target, i_target = self.target_encoder(inputs)  # Output from the target encoder
        return self.predictor(u_online), u_target, self.predictor(
            i_online), i_target  # Return transformed online representations and target representations

    @torch.no_grad()
    def get_embedding(self):
        """
        Get the online representations of all users and all items.

        Returns:
            tuple: Online representations of users and items, transformed online representations of users and items.
        """
        u_online, i_online = self.online_encoder.get_embedding()  # Get embeddings from the online encoder
        return self.predictor(u_online), u_online, self.predictor(
            i_online), i_online  # Return transformed online representations

    def get_loss(self, output):
        """
        Compute contrastive loss.

        Parameters:
            output (tuple): Online and target representations of users and items.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)  # Normalize online user representation
        u_target = F.normalize(u_target, dim=-1)  # Normalize target user representation
        i_online = F.normalize(i_online, dim=-1)  # Normalize online item representation
        i_target = F.normalize(i_target, dim=-1)  # Normalize target item representation

        # Euclidean distance can be replaced by negative inner product of normalized vectors
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)  # Contrastive loss for user-item pairs
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)  # Contrastive loss for item-user pairs

        return (loss_ui + loss_iu).mean()  # Return mean contrastive loss


class GAT_Encoder(nn.Module):
    def __init__(self, user_count, item_count, latent_size, edge_index, n_layers=3, drop_flag=False):
        super(GAT_Encoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.edge_index = edge_index  # Edge indices
        self.drop_ratio = 0.2  # Dropout ratio
        self.drop_flag = drop_flag  # Whether to enable dropout

        # Initialize user and item embeddings
        self.embedding_dict = self._init_model()

        # Build graph attention layers
        self.convs = nn.ModuleList([
            GATConv(latent_size, latent_size, heads=1, concat=True) for _ in range(n_layers)
        ])

    def _init_model(self):
        """
        Initialize user and item embeddings.
        """
        initializer = nn.init.xavier_uniform_  # Use Xavier initialization
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),  # User embeddings
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))  # Item embeddings
        })
        return embedding_dict

    def forward(self, inputs):
        """
        Forward pass function, compute the representations of users and items.

        Parameters:
            inputs (dict): A dictionary containing 'user' and 'item' keys corresponding to user IDs and item IDs.

        Returns:
            tuple: Representations of users and items.
        """
        # Concatenate user and item embeddings
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        # Perform multi-layer graph attention propagation
        for conv in self.convs:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            if self.drop_flag:
                ego_embeddings = F.dropout(ego_embeddings, p=self.drop_ratio, training=self.training)

        # Separate user and item embeddings
        user_all_embeddings = ego_embeddings[:self.user_count, :]
        item_all_embeddings = ego_embeddings[self.user_count:, :]

        # Get embeddings for input user and item IDs
        users, items = inputs['user'], inputs['item']
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings  # Return user and item representations

    @torch.no_grad()
    def get_embedding(self):
        """
        Get embeddings of all users and all items.

        Returns:
            tuple: Embeddings of all users and all items.
        """
        # Concatenate user and item embeddings
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        # Perform multi-layer graph attention propagation
        for conv in self.convs:
            ego_embeddings = conv(ego_embeddings, self.edge_index)

        # Separate user and item embeddings
        user_all_embeddings = ego_embeddings[:self.user_count, :]
        item_all_embeddings = ego_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings  # Return embeddings of all users and all items



