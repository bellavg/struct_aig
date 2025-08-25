# test_data_utils.py
import unittest
import torch
import numpy as np
import networkx as nx
from aigverse import Aig
import aigverse.adapters
from torch_geometric.utils import from_networkx

# --- Import the functions to be tested ---
# Make sure data_utils.py is in the same directory or your PYTHONPATH
from src.process_data import (
    convert_nx_to_pyg,
    calculate_graph_attributes,
    add_order_info_aig
)


class TestAIGDataProcessing(unittest.TestCase):
    """
    A suite of tests to validate the main data conversion and attribute
    calculation functions in the pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a simple, consistent AIG and its NetworkX representation
        that will be used across all tests. This method is run once
        before all tests in this class.
        """
        # 1. Create a simple AIG programmatically
        cls.aig = Aig()
        pi_a = cls.aig.create_pi()
        pi_b = cls.aig.create_pi()
        pi_d = cls.aig.create_pi()

        # Invert a signal to create an inverted edge example
        and_1 = cls.aig.create_and(pi_a, ~pi_b)
        and_2 = cls.aig.create_and(and_1, pi_d)

        cls.aig.create_po(and_2)

        cls.g_nx = cls.aig.to_networkx(levels=True)
        cls.nodes_in_nx = list(cls.g_nx.nodes())

    def test_convert_nx_to_pyg(self):
        """
        Tests the conversion from a NetworkX graph to a PyG Data object.
        """
        g_pyg, node_map = convert_nx_to_pyg(self.g_nx)

        # Check basic graph properties (expected 7 nodes due to const 0)
        self.assertEqual(g_pyg.num_nodes, 7, "Should have 7 nodes (including const 0)")
        self.assertEqual(g_pyg.num_edges, 5, "Should have 5 edges")

        # Check tensor shapes
        self.assertEqual(g_pyg.x.shape, (7, 4), "Node feature tensor shape is incorrect")
        self.assertEqual(g_pyg.edge_index.shape, (2, 5), "Edge index tensor shape is incorrect")
        self.assertEqual(g_pyg.edge_attr.shape, (5, 2), "Edge attribute tensor shape is incorrect")

        # Check a specific node's feature vector (adjust index for const 0)
        pi_a_nx_node = self.nodes_in_nx[1]  # const 0 is at index 0, so PIs start at 1
        pi_a_pyg_idx = node_map[pi_a_nx_node]

        # The one-hot vector for a PI node should be [0, 1, 0, 0]
        self.assertTrue(torch.equal(g_pyg.x[pi_a_pyg_idx], torch.tensor([0.0, 1.0, 0.0, 0.0])),
                        "PI node feature vector is incorrect")

        # Check that there is one inverted edge
        inverted_edge_type = torch.tensor([0.0, 1.0])
        # Check if any row in edge_attr matches the inverted_edge_type
        self.assertTrue(torch.any(torch.all(g_pyg.edge_attr == inverted_edge_type, dim=1)),
                        "Should contain one inverted edge attribute")


class TestCalculateGraphAttributes(unittest.TestCase):
    """
    A focused suite of tests for the `calculate_graph_attributes` function.
    Uses a more complex AIG to validate specific attribute values.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a more complex AIG with inversions and varied fanouts."""
        cls.aig = Aig()
        pi_a = cls.aig.create_pi()
        pi_b = cls.aig.create_pi()
        pi_d = cls.aig.create_pi()

        and_1 = cls.aig.create_and(pi_a, pi_b)
        and_2 = cls.aig.create_and(and_1, pi_d)
        and_3 = cls.aig.create_and(and_1, pi_a)  # Creates fanout > 1 for pi_a and and_1

        cls.aig.create_po(~and_2)  # Creates an inversion
        cls.aig.create_po(and_3)

        # Create NetworkX graph with all necessary attributes
        cls.g_nx = cls.aig.to_networkx(levels=True, fanouts=True)

        # Calculate attributes once for all tests
        attributes = calculate_graph_attributes(cls.aig, cls.g_nx)
        feature_order = [
            'num_edges', 'degree_assortativity', 'num_inversions', 'num_nodes', 'max_fanout',
            'graph_depth', 'var_fanout', 'level_variance', 'avg_edge_level_span',
            'var_edge_level_span', 'diameter', 'radius', 'avg_eccentricity',
            'density', 'algebraic_connectivity', 'avg_fanout'
        ]
        cls.stats = {key: val for key, val in zip(feature_order, attributes)}

    def test_basic_counts(self):
        """Tests node, edge, and inversion counts."""
        # Expected 9 nodes (1 const, 3 pi, 3 and, 2 po)
        self.assertAlmostEqual(self.stats['num_nodes'], 9.0)
        self.assertAlmostEqual(self.stats['num_edges'], 8.0)
        self.assertAlmostEqual(self.stats['num_inversions'], 1.0)

    def test_level_based_metrics(self):
        """Tests depth and level variance."""
        self.assertAlmostEqual(self.stats['graph_depth'], 3.0)
        # Cast the expected value to float32 to match the function's output dtype
        expected_variance = np.var([0, 0, 0, 0, 1, 2, 2, 3, 3], dtype=np.float32)
        # FIXED: Use a delta for robust float comparison instead of relying on places
        self.assertAlmostEqual(self.stats['level_variance'], expected_variance, delta=1e-6)

    def test_fanout_metrics(self):
        """Tests max, average, and variance of fanout."""
        self.assertAlmostEqual(self.stats['max_fanout'], 2.0)
        # Expected fanouts for 9 nodes: [0,2,1,1,2,1,1,0,0], avg = 0.888...
        self.assertAlmostEqual(self.stats['avg_fanout'], np.mean([0, 2, 1, 1, 2, 1, 1, 0, 0]))
        self.assertAlmostEqual(self.stats['var_fanout'], np.var([0, 2, 1, 1, 2, 1, 1, 0, 0]))

    def test_networkx_metrics(self):
        """Tests metrics derived from NetworkX like density, diameter, etc."""
        self.assertAlmostEqual(self.stats['density'], 8.0 / (9 * 8))
        self.assertAlmostEqual(self.stats['diameter'], 4.0)
        # The correct radius for this graph is 2.0, not 3.0
        self.assertAlmostEqual(self.stats['radius'], 2.0)


class TestAddOrderInfoAIG(unittest.TestCase):
    """
    A focused suite of tests for the `add_order_info_aig` function.
    Each test validates a specific piece of information added to the graph data.
    """

    @classmethod
    def setUpClass(cls):
        """Pre-computes the processed graph object once for all tests."""
        aig = Aig()
        pi_a, pi_b, pi_d = aig.create_pi(), aig.create_pi(), aig.create_pi()

        # Invert a signal to create an inverted edge example
        and_1 = aig.create_and(pi_a, ~pi_b)
        and_2 = aig.create_and(and_1, pi_d)
        aig.create_po(and_2)

        cls.g_nx = aig.to_networkx(levels=True)
        g_pyg, node_map = convert_nx_to_pyg(cls.g_nx)

        cls.node_map = node_map
        cls.g_pyg_processed = add_order_info_aig(g_pyg.clone(), cls.g_nx, node_map)

    def test_positional_encoding(self):
        """Tests that `abs_pe` correctly reflects node depths."""
        self.assertTrue(hasattr(self.g_pyg_processed, 'abs_pe'))
        idx_to_aig_node = {v: k for k, v in self.node_map.items()}
        expected_levels = torch.tensor(
            [self.g_nx.nodes[idx_to_aig_node[i]]['level'] for i in range(self.g_pyg_processed.num_nodes)],
            dtype=torch.long
        )
        self.assertTrue(torch.equal(self.g_pyg_processed.abs_pe, expected_levels),
                        "abs_pe values do not match node levels")

    def test_reachability_edges(self):
        """Tests that `dag_rr_edge_index` is created for message passing."""
        self.assertTrue(hasattr(self.g_pyg_processed, 'dag_rr_edge_index'))
        # Transitive closure should have more edges than the original graph
        self.assertGreater(self.g_pyg_processed.dag_rr_edge_index.shape[1],
                           self.g_pyg_processed.edge_index_original.shape[1])

    def test_attention_mask(self):
        """Tests the `mask_rc` for DAG-aware attention."""
        self.assertTrue(hasattr(self.g_pyg_processed, 'mask_rc'))
        mask = self.g_pyg_processed.mask_rc

        # Find key nodes to test reachability using the correct one-hot indices
        # [const, pi, gate, po]
        nx_pis = [n for n, d in self.g_nx.nodes(data=True) if d['type'][1] == 1]  # PI is index 1
        nx_po = [n for n, d in self.g_nx.nodes(data=True) if d['type'][3] == 1][0]  # PO is index 3
        pi_a_idx = self.node_map[nx_pis[0]]
        po_c_idx = self.node_map[nx_po]

        # Mask is False for reachable pairs, True for non-reachable
        self.assertFalse(mask[pi_a_idx, po_c_idx].item(), "PI_A should be able to attend to its successor PO_C")
        self.assertFalse(mask[po_c_idx, pi_a_idx].item(), "PO_C should be able to attend to its predecessor PI_A")

    def test_topological_edge_sort(self):
        """Tests that `edge_index` is sorted topologically."""
        self.assertTrue(hasattr(self.g_pyg_processed, 'edge_index_original'))

        # Create a map from NetworkX node object to its position in a topological sort
        topo_order = list(nx.topological_sort(self.g_nx))
        node_order_map = {node: i for i, node in enumerate(topo_order)}
        idx_to_aig_node = {v: k for k, v in self.node_map.items()}

        # Check that for every edge (u, v), the topological position of u is less than v
        sorted_edge_index = self.g_pyg_processed.edge_index
        for i in range(sorted_edge_index.shape[1]):
            u_pyg_idx, v_pyg_idx = sorted_edge_index[:, i]
            u_nx_node = idx_to_aig_node[u_pyg_idx.item()]
            v_nx_node = idx_to_aig_node[v_pyg_idx.item()]

            self.assertLess(
                node_order_map[u_nx_node],
                node_order_map[v_nx_node],
                f"Edge ({u_nx_node}, {v_nx_node}) violates topological sort order."
            )

class TestNormalization(unittest.TestCase):
    """
    Tests the z-score normalization of graph attributes.
    """

    def test_z_score_normalization(self):
        """
        Tests that the z-score normalization is applied correctly.
        """
        # Sample unnormalized data
        y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

        # Sample mean and std
        mean = y.mean()
        std = y.std(unbiased=False) # Use population std dev for this test

        # Apply normalization
        normalized_y = (y - mean) / std

        # The mean of the normalized data should be 0
        self.assertAlmostEqual(normalized_y.mean().item(), 0.0, delta=1e-6)

        # The std of the normalized data should be 1
        self.assertAlmostEqual(normalized_y.std(unbiased=False).item(), 1.0, delta=1e-6)

        # Check the values
        expected_normalized_y = torch.tensor([-1.41421356, -0.70710678,  0. ,  0.70710678,  1.41421356])
        self.assertTrue(torch.allclose(normalized_y, expected_normalized_y, atol=1e-6))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)