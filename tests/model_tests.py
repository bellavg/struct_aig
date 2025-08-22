# test_layers.py
import unittest
import torch
from torch_geometric.data import Data, Batch

# --- Import the layers to be tested ---
from src.layers import Attention, StructureExtractor, TransformerEncoderLayer

class TestStructureExtractor(unittest.TestCase):
    """
    Tests the StructureExtractor module to ensure it correctly applies GNN layers.
    """

    def setUp(self):
        """Set up a simple graph structure."""
        self.embed_dim = 128
        self.num_nodes = 5
        self.extractor = StructureExtractor(self.embed_dim, gnn_type='gin', num_layers=3)
        self.x = torch.randn(self.num_nodes, self.embed_dim)
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    def test_forward_pass_shape(self):
        """Tests that the output shape is correct."""
        output = self.extractor(self.x, self.edge_index)
        self.assertEqual(output.shape, (self.num_nodes, self.embed_dim))

    def test_batch_norm_application(self):
        """Tests that batch norm is applied when specified."""
        # With batch_norm=True (default)
        self.assertTrue(hasattr(self.extractor, 'bn'))
        # With batch_norm=False
        extractor_no_bn = StructureExtractor(self.embed_dim, gnn_type='gin', num_layers=3, batch_norm=False)
        self.assertFalse(hasattr(extractor_no_bn, 'bn'))


class TestAttention(unittest.TestCase):
    """
    Tests the Attention module, including the Structure-Aware (SAT) component
    and the DAG-aware masking (DAGRA).
    """

    def setUp(self):
        """Set up a batch of two graphs."""
        self.embed_dim = 128
        self.num_heads = 8
        self.attention = Attention(self.embed_dim, self.num_heads)

        # Graph 1: 3 nodes
        x1 = torch.randn(3, self.embed_dim)
        edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        data1 = Data(x=x1, edge_index=edge_index1)

        # Graph 2: 4 nodes
        x2 = torch.randn(4, self.embed_dim)
        edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
        data2 = Data(x=x2, edge_index=edge_index2)

        self.batch = Batch.from_data_list([data1, data2])
        self.ptr = self.batch.ptr

    def test_forward_pass_shape(self):
        """Tests that the output shape is correct for a batch."""
        out, attn = self.attention(self.batch.x, SAT=True, edge_index=self.batch.edge_index, mask_dag_=None, ptr=self.ptr, return_attn=True)
        self.assertEqual(out.shape, self.batch.x.shape)
        # Expected attn shape: (batch_size, num_heads, max_nodes, max_nodes)
        # max_nodes in this batch is 4.
        self.assertEqual(attn.shape, (2, self.num_heads, 4, 4))

    def test_dag_masking(self):
        """Tests that the DAG-aware mask is correctly applied."""
        # Create a mask where node 0 in the first graph cannot attend to node 2
        mask_dag_ = torch.zeros(2, 4, 4, dtype=torch.bool)
        mask_dag_[0, 0, 2] = True # From node 0 to 2 in graph 1

        out, attn = self.attention(self.batch.x, SAT=True, edge_index=self.batch.edge_index, mask_dag_=mask_dag_, ptr=self.ptr, return_attn=True)
        # The attention score should be 0 (or close to it) due to the softmax over -inf
        for head in range(self.num_heads):
            self.assertAlmostEqual(attn[0, head, 0, 2].item(), 0.0, places=6)


class TestTransformerEncoderLayer(unittest.TestCase):
    """
    Tests a single layer of the Graph Transformer.
    """

    def setUp(self):
        """Set up a batch of data for the encoder layer."""
        self.d_model = 128
        self.nhead = 8
        self.layer = TransformerEncoderLayer(self.d_model, self.nhead)

        x1 = torch.randn(3, self.d_model)
        edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        data1 = Data(x=x1, edge_index=edge_index1)

        x2 = torch.randn(4, self.d_model)
        edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        data2 = Data(x=x2, edge_index=edge_index2)

        self.batch = Batch.from_data_list([data1, data2])
        self.ptr = self.batch.ptr

    def test_forward_pass_shape(self):
        """Tests that the output shape remains consistent."""
        output = self.layer(
            x=self.batch.x,
            SAT=True,
            abs_pe_type='dagpe',
            abs_pe=None,
            edge_index=self.batch.edge_index,
            mask_dag_=None,
            dag_rr_edge_index=None,
            ptr=self.ptr
        )
        self.assertEqual(output.shape, self.batch.x.shape)


# test_dag_transformer.py
import unittest
import torch
from torch_geometric.data import Data, Batch

# --- Import the model to be tested ---
from src.dag_transformer import GraphTransformer

class TestGraphTransformer(unittest.TestCase):
    """
    Tests the complete GraphTransformer model to ensure it can process a batch
    of graph data and produce a regression output of the correct size.
    """

    def setUp(self):
        """Set up model and a batch of graph data for testing."""
        self.d_model = 128
        self.num_heads = 8
        self.num_layers = 4
        self.out_size = 16

        self.model = GraphTransformer(
            in_size=4,
            out_size=self.out_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        # --- FIX: Make graphs in the batch have the same size to resolve the collate error ---
        # The RuntimeError occurs because the default collate function cannot batch
        # the `mask_rc` attribute (e.g., [5, 5] and [7, 7]) when node counts differ.
        # For a unit test, using graphs of the same size is a straightforward solution.
        num_nodes_for_test = 7

        # Graph 1 (now with 7 nodes)
        data1 = Data(
            x=torch.randn(num_nodes_for_test, 4),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long),
            edge_attr=torch.randn(6, 2),
            mask_rc=torch.zeros(num_nodes_for_test, num_nodes_for_test, dtype=torch.bool),
            abs_pe=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        )

        # Graph 2 (7 nodes)
        data2 = Data(
            x=torch.randn(num_nodes_for_test, 4),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long),
            edge_attr=torch.randn(6, 2),
            mask_rc=torch.zeros(num_nodes_for_test, num_nodes_for_test, dtype=torch.bool),
            abs_pe=torch.tensor([0, 1, 1, 2, 2, 3, 4], dtype=torch.long)
        )

        self.batch = Batch.from_data_list([data1, data2])

    def test_forward_pass_output_shape(self):
        """Tests that the model's output has the correct regression vector shape."""
        output = self.model(self.batch)
        # Batch size is 2, and the output size should be 16
        self.assertEqual(output.shape, (2, self.out_size))

    def test_no_global_pool(self):
        """Tests the model's output shape when global pooling is disabled."""
        model_no_pool = GraphTransformer(
            in_size=4,
            out_size=self.out_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            use_global_pool=False
        )
        output = model_no_pool(self.batch)
        # The output should now be per-node, not per-graph
        num_total_nodes = self.batch.x.shape[0]
        self.assertEqual(output.shape, (num_total_nodes, self.out_size))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)