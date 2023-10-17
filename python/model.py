'''
filename: model.py
Author: Dario Pullia
Date created: 04/10/2023

Description:




'''
import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
torch.manual_seed(42)


class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        first_layer_neurons = model_params["model_first_layer_neurons"]

        self.first_layer = Linear(feature_size, first_layer_neurons)
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.conv1 = GCNConv(first_layer_neurons, embedding_size)
        self.transf1 = Linear(embedding_size, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(GCNConv(embedding_size, embedding_size))
            self.transf_layers.append(Linear(embedding_size, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

        self.linear1 = Linear(2 * embedding_size, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 10)

    def forward(self, x, edge_index, batch_index):

        x = self.first_layer(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.transf1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        for i in range(self.n_layers):
            x = F.relu(self.conv_layers[i](x, edge_index))
            x = self.bn_layers[i](x)
            x = self.transf_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Replace Top-K Pooling with Global Mean Pooling
        x = global_mean_pool(x, batch_index)

        x = torch.cat([x, x], dim=1)  # Concatenating with itself for simplicity
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x