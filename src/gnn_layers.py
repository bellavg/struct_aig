# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch.nn import Sequential, Linear, ReLU

# This list is imported by layers.py, so it must be kept.
EDGE_GNN_TYPES = [
    'gine', 'gcn',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4'
]

def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    """
    A simplified factory function that now only returns a GIN layer,
    as required by the StructureExtractor.
    """
    if gnn_type == "gin":
        # A multi-layer perceptron (MLP) that defines the behavior of the GIN layer.
        mlp = Sequential(
            Linear(embed_dim, embed_dim),
            ReLU(True),
            Linear(embed_dim, embed_dim),
        )
        # The Graph Isomorphism Network (GIN) convolutional layer.
        return gnn.GINConv(mlp, train_eps=True)
    else:
        # This error will be raised if the StructureExtractor is ever changed
        # to a GNN type that has been removed.
        raise ValueError(f"Unsupported GNN type: {gnn_type}. Only 'gin' is supported in this version.")
