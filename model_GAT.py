import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np

from torch_geometric.nn import Sequential, GATConv, TransformerConv
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn.models import InnerProductDecoder, GAE, VGAE
from torch_geometric.nn import GATConv, GAE


# Define GAT-based encoder for GAE
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.6)
        self.conv2 = GATConv(32 * 1, out_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize GAE model with GAT encoder and move it to the GPU
class GAEModel(GAE):
    def __init__(self, in_channels, out_channels):
        encoder = GATEncoder(in_channels, out_channels)
        super(GAEModel, self).__init__(encoder)

    def get_attention_scores(self, data):
        x, edge_index = data.x, data.edge_index
        # Pass data through the first GAT layer to get attention scores
        _, (edge_index_selfloop, alpha) = self.encoder.conv1(x, edge_index, return_attention_weights=True)
        # matrix shape: number of edges x number of heads
        return edge_index_selfloop,alpha
    

class Encoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear1 = Linear(dim, dim)
        self.linear2 = Linear(dim, dim)

        # self loop is default，so also include the attention of self-loop
        # delete self loop
        self.conv1 = GATConv(dim, dim, add_self_loops=False)
        self.conv2 = GATConv(dim, dim, add_self_loops=False)
        
        # self.conv1 = TransformerConv(dim, dim, heads=1)
        # self.conv2 = TransformerConv(dim, dim, heads=1)

        self.act = torch.nn.CELU()

    def cat(self, x_gene, x_cell, y):
        result = []
        count_gene = 0
        count_cell = 0

        for i in y:
            if i:
                result.append(x_gene[count_gene].view(1, -1))
                count_gene += 1
            else:
                result.append(x_cell[count_cell].view(1, -1))
                count_cell += 1

        result = torch.cat(result)
        return result

    def forward(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y

        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return x

    def get_att(self, graph):
        x, edge_index,  y = graph.x, graph.edge_index, graph.y
        print(x.shape,y.shape)
        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x, att = self.conv2(x, edge_index, return_attention_weights=True)

        return x, att


class SenGAE(GAE):
    def __init__(self):
        super(SenGAE, self).__init__(encoder=Encoder(),
                                     decoder=InnerProductDecoder())

    def forward(self, graph, split=10):
        z = self.encode(graph)
        # adj_pred = self.decoder(z)
        return z

