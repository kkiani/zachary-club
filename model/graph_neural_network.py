import torch as th
import torch.nn as nn

from dgl import DGLGraph
from dgl.nn import GraphConv


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphConvolutionNetwork, self).__init__()

        self.gcl1 = GraphConv(in_feats, hidden_size)
        self.gcl2 = GraphConv(hidden_size, num_classes)

    def forward(self, graph: DGLGraph, inputs):
        x = self.gcl1(graph, inputs)
        x = th.relu(x)
        x = self.gcl2(graph, x)
        
        return nn.Softmax(x)