import os, pickle, argparse
from torch.autograd import grad
import dgl
import torch as th
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from dgl.nn import GraphConv


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphConvolutionNetwork, self).__init__()

        self.gcl1 = GraphConv(in_feats, hidden_size)
        self.gcl2 = GraphConv(hidden_size, num_classes)

    def forward(self, graph: dgl.DGLGraph, inputs):
        x = self.gcl1(graph, input)
        x = th.relu(x)
        x = self.gcl2(graph, x)
        
        return th.softmax(x)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--node-count', type=int)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    args, _ = parser.parse_known_args()

    # Setup Variables
    node_count = args.node_count
    epochs = args.epochs
    learning_rate = args.learning_rate

    training_dir = os.environ['SM_CHANNEL_TRAINING']
    model_dir    = os.environ['SM_MODEL_DIR']

    # Loading Dataset
    edge_list = []
    with open(os.path.join(training_dir, 'edge_list.pickle'), 'rb') as file:
        edge_list = pickle.load(file)
        edge_list = [[int(x[0]), int(x[1])] for x in edge_list]     # Dataset is in str format and dgl supports only int datatypes.

    # Building Graph
    assert edge_list, "Not able to load the dataset."
    graph = dgl.DGLGraph()
    graph.add_nodes(node_count)
    src, dst = tuple(zip(*edge_list))
    graph.add_edges(src, dst)
    graph.add_edges(dst, src)

    print(f'The model have {graph.number_of_nodes()} nodes.')
    print(f'The model have {graph.number_of_edges()} edges.')

    # Building GCN Model
    model = GraphConvolutionNetwork(node_count, 5, 2)

    # Training Model
    inputs = th.eye(node_count)
    labeled_nodes = th.tensor([0, node_count-1])
    label = th.tensor([0, 1])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    all_pred = []

    for epoch in epochs:
        pred = model(graph, inputs)
        all_pred.append(pred)

        loss = F.cross_entropy(pred[labeled_nodes], label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    last_epoch = all_pred[epochs-1].detach().numpy()
    predicted_class = np.argmax(last_epoch, axis=-1)

    print(predicted_class)

    th.save(model.state_dict(), os.path.join(model_dir, 'karate_club.pt'))

if __name__ == '__main__':
    main()