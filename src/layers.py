# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from src.utils import pad_batch, unpad_batch
import torch.nn.functional as F

#

from src.gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES


class Attention(nn.Module):
    """
    The core attention mechanism for the Graph Transformer.
    Includes the Structure-Aware (SAT) component and the DAG-aware masking (DAGRA).
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.batch_first = True
        self._qkv_same_embed_dim = True
        # --- FIX: Add the in_proj_bias attribute and set to None ---
        self.in_proj_bias = None
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # The GNN-based structure extractor for the SAT component.
        self.structure_extractor = StructureExtractor(embed_dim, gnn_type='gin', **kwargs)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)
        # --- FIX: Since to_qk handles both Q and K, its bias is the in_proj_bias ---
        # Although we set it to None above for the check, we can link it here.
        # This is more for completeness, the main fix is setting it to None initially.
        self.in_proj_bias = self.to_qk.bias


    def forward(self, x, SAT, edge_index, mask_dag_,
                edge_attr=None, ptr=None, return_attn=False):

        # 1. Create the value matrix from the input features.
        v = self.to_v(x)

        # 2. Create the query and key matrices.
        # This is where the "Structure-Aware" part happens.
        x_struct = self.structure_extractor(x, edge_index, edge_attr)
        qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # 3. Compute self-attention using the DAG-aware mask.
        out, attn = self.self_attn(qk, v, ptr, mask_dag_, return_attn=return_attn)
        return self.out_proj(out), attn

    # In src/layers.py, inside the Attention class

    def self_attn(self, qk, v, ptr, mask_dag_list, return_attn=False):
        """ Self-attention with padding and DAG reachability mask. """
        # This part is the same as the original: pad the features for the batch.
        # It also gives us the padding_mask, which tells us the target shape.
        qk_padded, padding_mask = pad_batch(qk, ptr, return_mask=True)

        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk_padded)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # This is also the same: apply the standard mask to ignore padded nodes.
        dots = dots.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # --- CHANGE: This block replaces the original's fixed-size mask logic ---
        # It adapts the original idea for your variable-sized graphs.
        if mask_dag_list is not None:
            bsz, max_nodes = padding_mask.shape

            # 1. Create a new, empty, dense mask for the entire batch.
            #    It's perfectly sized to match the padded 'dots' tensor.
            dag_mask = torch.ones(bsz, max_nodes, max_nodes, device=dots.device, dtype=torch.bool)

            # 2. Loop through the list of unpadded masks and place them into the dense mask.
            for i, mask in enumerate(mask_dag_list):
                num_nodes = mask.shape[0]
                dag_mask[i, :num_nodes, :num_nodes] = mask

            # 3. Apply the correctly aligned DAG mask.
            dots = dots.masked_fill(dag_mask.unsqueeze(1), float('-inf'))
        # --- End of changes ---

        dots = dots + 1e-9
        dots = torch.clamp(dots, min=-1e9, max=1e9)  # Clamp values to a reasonable range

        dots = self.attend(dots)
        dots = self.attn_dropout(dots)

        # This part is the same as the original: pad v, compute output, and unpad.
        v_padded = pad_batch(v, ptr)
        v_padded = rearrange(v_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v_padded)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        return (out, dots) if return_attn else (out, None)


class StructureExtractor(nn.Module):
    """
    A GNN-based module to extract local structural information for each node,
    which is the core of the Structure-Aware Transformer (SAT).
    """

    def __init__(self, embed_dim, gnn_type="gin", num_layers=3, batch_norm=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        # self.batch_norm = batch_norm
        # if batch_norm:
        self.bn = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None):
        for gcn_layer in self.gcn:
            if self.gnn_type in EDGE_GNN_TYPES:
                x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

        if self.num_layers > 0:
            x = self.bn(x)

        x = self.out_proj(x)
        return x


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    A single layer of the Graph Transformer. It combines the DAG-aware
    self-attention mechanism with a standard feed-forward network.
    """

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout, bias=True, **kwargs)
        # self.batch_norm = batch_norm
        # if batch_norm:
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, SAT, abs_pe_type, abs_pe, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, degree=None, ptr=None, return_attn=False):
        normed_x = self.norm1(x)

        # Self-Attention block
        x2, attn = self.self_attn(
            x=normed_x,
            SAT=SAT,
            edge_index=edge_index,
            mask_dag_=mask_dag_,
            edge_attr=edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )
        x = x + self.dropout1(x2)
        normed_x = self.norm2(x)

        # Feed-Forward block
        x2 = self.linear2(self.dropout(self.activation(self.linear1(normed_x))))
        x = x + self.dropout2(x2)


        return x
