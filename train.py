#!/usr/bin/env python

import os, pickle, argparse
import dgl
import numpy as np
import torch as th
from torch import optim
import torch.nn.functional as F

from model.graph_neural_network import GraphConvolutionNetwork


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
    all_preds = []

    for epoch in range(epochs):
        preds = model(graph, inputs)
        all_preds.append(preds)

        loss = F.cross_entropy(preds[labeled_nodes], label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    last_epoch = all_preds[epochs-1].detach().numpy()
    predicted_class = np.argmax(last_epoch, axis=-1)

    print(predicted_class)

    th.save(model.state_dict(), os.path.join(model_dir, 'karate_club.pt'))

if __name__ == '__main__':
    main()