import torch
from torch import nn
import torch_geometric.nn as gnn
from src.layers import TransformerEncoderLayer
import math
from torch_geometric.utils import *



class GraphTransformerEncoder(nn.TransformerEncoder):
    """ A standard Transformer Encoder that is aware of DAG-specific arguments. """

    def forward(self, x, SAT, abs_pe, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, degree=None, ptr=None, return_attn=False):
        output = x
        for mod in self.layers:
            # Pass all the necessary arguments to each layer
            output = mod(output, SAT, 'dagpe', abs_pe, edge_index, mask_dag_, dag_rr_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         ptr=ptr,
                         return_attn=return_attn
                         )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    """
    A Graph Transformer tailored for DAGs, adapted for graph-level regression.
    It uses Structure-Aware Attention (SAT), depth-based positional encodings (DAGPE),
    and reachability-based attention (DAGRA) to predict a 16-dimensional structural attribute vector.
    """

    def __init__(self, in_size=4, out_size=1, d_model=128, num_heads=8,
                 dim_feedforward=512, dropout=0.1, num_layers=4,
                 batch_norm=True, num_edge_features=2,
                 in_embed=True, edge_embed=True, use_global_pool=True,
                 global_pool='mean', SAT=True, **kwargs):
        super().__init__()

        self.SAT = SAT

        # --- Positional Encoding Setup (DAGPE) ---
        # Pre-calculates sinusoidal embeddings for node depth.
        self.position = torch.arange(1000).unsqueeze(1)  # Increased range for deeper graphs
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.poe = torch.zeros(1000, d_model)
        self.poe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.poe[:, 1::2] = torch.cos(self.position * self.div_term)
        self.dropout = nn.Dropout(dropout)

        # --- Node and Edge Embeddings ---
        self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=False)

        # Edge embedding is now always active
        edge_dim = kwargs.get('edge_dim', 32)
        self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=edge_dim, bias=False)
        kwargs['edge_dim'] = edge_dim  # Ensure edge_dim is passed to the encoder layers

        # --- Transformer Encoder ---
        # The core of the model, built from our custom DAG-aware layers.
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        # --- Output Layers for Regression ---
        self.use_global_pool = use_global_pool
        if self.use_global_pool:
            self.pooling = gnn.global_mean_pool

        # Maps the aggregated graph representation to the 16-dimensional attribute vector
        self.regressor_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, out_size)
        )

    def forward(self, data, return_attn=False):
        # Deconstruct the graph data object
        # --- CHANGE: Get the list of masks from `mask_rc_list` ---
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        mask_dag_list = data.mask_rc_list

        # print(f"--- New Batch ---")
        # print(f"Initial x shape: {x.shape}")

        # Create a pointer for batching
        data.ptr = torch.cat([data.batch.new_zeros(1), torch.bincount(data.batch).cumsum(dim=0)], dim=0)

        # Get the reachability edge index for message-passing attention
        dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None

        # 1. Initial Node Embedding
        output = self.embedding(x)

        # 2. Add Positional Encoding (DAGPE)
        self.poe = self.poe.to(x.device)
        abs_pe = data.abs_pe
        abs_pe = torch.clamp(abs_pe, 0, self.poe.shape[0] - 1)
        pe = self.poe[abs_pe]
        output = output + pe
        output = self.dropout(output)

        # 3. Edge Embedding
        if edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
        else:
            edge_attr = None

        # 4. Pass through the Transformer Encoder

        output = self.encoder(
            output,
            SAT=self.SAT,
            abs_pe=abs_pe,
            edge_index=edge_index,
            mask_dag_=mask_dag_list,  # This list is now passed to the attention layer
            dag_rr_edge_index=dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )

        # 5. Global Pooling
        if self.use_global_pool:
            output = self.pooling(output, batch)

        # 6. Final Regression Output
        output = self.regressor_head(output)

        return output