# -*- coding: utf-8 -*-
from __future__ import print_function
import gzip
import pickle
import numpy as np
import torch
from tqdm import tqdm
import os
import argparse
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, from_networkx

# --- NEW: Import aigverse and its views ---
import aigverse
from typing import Any, Final

import networkx as nx
import numpy as np

from aigverse import Aig, DepthAig, simulate, simulate_nodes, to_edge_list


def to_networkx(
    self: Aig,
    *,
    levels: bool = False,
    fanouts: bool = False,
    node_tts: bool = False,
    graph_tts: bool = False,
    dtype: type[np.generic] = np.int8,
) -> nx.DiGraph:
    """Converts an :class:`~aigverse.Aig` to a :class:`~networkx.DiGraph`.

    This function transforms the AIG into a directed graph representation
    using the NetworkX library. It allows for the inclusion of various
    attributes for the graph, its nodes, and edges, making it suitable
    for graph-based machine learning tasks.

    Note that the constant-0 node is always included in the graph, as
    index 0, even if it is not referenced by any edges.

    Args:
        self: The AIG object to convert.
        levels: If True, computes and adds level information for each node
            and the total number of levels to the graph, as attributes
            ``levels`` and ``level``, respectively. Defaults to False.
        fanouts: If True, adds fanout size information for each node
            as a ``fanouts`` attribute. Defaults to False.
        node_tts: If True, computes and adds a truth table for each node
            as a ``function`` attribute. Defaults to False.
        graph_tts: If True, computes and adds the graph's overall truth
            table as a ``function`` attribute to the graph. Defaults to False.
        dtype: The data type for truth tables and all one-hot encodings.
            Defaults to :obj:`~numpy.int8`. For machine learning tasks, a
            floating-point type such as :obj:`~numpy.float32` or
            :obj:`~numpy.float64` may be more appropriate, as it allows
            for gradient-based optimization.

    Returns:
        A :class:`~networkx.DiGraph` representing the AIG.

    Graph Attributes:
        - type (str): ``"AIG"``.
        - num_pis (int): Number of primary inputs.
        - num_pos (int): Number of primary outputs.
        - num_gates (int): Number of AND gates.
        - levels (int, optional): Total number of levels in the AIG.
        - function (list[:class:`~numpy.ndarray`], optional): Graph's truth tables.

    Node Attributes:
        - index (int): The node's identifier.
        - level (int, optional): The level of the node in the AIG.
        - function (:class:`~numpy.ndarray`, optional): The node's truth table.
        - type (:class:`~numpy.ndarray`): A one-hot encoded vector representing
            the node type (``[const, pi, gate, po]``). The data type is determined
            by the ``dtype`` argument, defaulting to :obj:`~numpy.int8`.

    Edge Attributes:
        - type (:class:`~numpy.ndarray`): A one-hot encoded vector representing the edge
            type (``[regular, inverted]``). The data type is determined by the
            ``dtype`` argument, defaulting to :obj:`~numpy.int8`.
    """
    # one-hot encodings for node types: [const, pi, gate, po]
    node_type_const: Final[np.ndarray[Any, np.dtype[np.int8]]] = np.array([1, 0, 0, 0], dtype=dtype)
    node_type_pi: Final[np.ndarray[Any, np.dtype[np.int8]]] = np.array([0, 1, 0, 0], dtype=dtype)
    node_type_gate: Final[np.ndarray[Any, np.dtype[np.int8]]] = np.array([0, 0, 1, 0], dtype=dtype)
    node_type_po: Final[np.ndarray[Any, np.dtype[np.int8]]] = np.array([0, 0, 0, 1], dtype=dtype)

    # one-hot encodings for edge types: [regular, inverted]
    edge_type_regular: Final[np.ndarray[Any, np.dtype[np.int8]]] = np.array([1, 0], dtype=dtype)
    edge_type_inverted: Final[np.ndarray[Any, np.dtype[np.int8]]] = np.array([0, 1], dtype=dtype)

    # Conditionally compute levels if requested
    if levels:
        depth_aig = DepthAig(self)

    node_funcs = {}
    graph_funcs = []

    # Conditionally compute node truth tables if requested
    if node_tts:
        node_funcs = {node: np.array(tt, dtype=dtype) for node, tt in simulate_nodes(self).items()}
        graph_funcs = [np.array(tt, dtype=dtype) for tt in simulate(self)]

    # Conditionally compute graph output truth tables if requested
    elif graph_tts:
        graph_funcs = [np.array(tt, dtype=dtype) for tt in simulate(self)]

    # Initialize the networkx graph
    g = nx.DiGraph()

    # Add global graph attributes
    g.graph["type"] = "AIG"
    g.graph["num_pis"] = self.num_pis()
    g.graph["num_pos"] = self.num_pos()
    g.graph["num_gates"] = self.num_gates()
    if levels:
        g.graph["levels"] = depth_aig.num_levels() + 1  # + 1 for the PO level
    if graph_tts:
        g.graph["function"] = graph_funcs

    # Iterate over all nodes in the AIG, plus synthetic PO nodes
    for node in self.nodes() + [self.po_index(po) + self.size() for po in self.pos()]:
        # Prepare node attributes dictionary
        attrs: dict[str, Any] = {"index": node}
        is_synthetic_po = node >= self.size()  # type: ignore[operator]

        if is_synthetic_po:
            type_vec = node_type_po
            if levels:
                attrs["level"] = depth_aig.num_levels() + 1
            if fanouts:
                attrs["fanouts"] = 0
            if node_tts:
                po_index = self.node_to_index(node) - self.size()
                attrs["function"] = graph_funcs[po_index]
        else:  # regular node
            if self.is_constant(node):
                type_vec = node_type_const
            elif self.is_pi(node):
                type_vec = node_type_pi
            else:  # is gate
                type_vec = node_type_gate

            if levels:
                attrs["level"] = depth_aig.level(node)
            if fanouts:
                attrs["fanouts"] = self.fanout_size(node)
            if node_tts:
                attrs["function"] = node_funcs[self.node_to_index(node)]

        attrs["type"] = type_vec

        # Add the node to the graph with its attributes
        g.add_node(node, **attrs)

    # Export the AIG as an edge list
    edges = to_edge_list(self)

    # Iterate over all edges and add them to the graph
    for src, tgt, weight in [(e.source, e.target, e.weight) for e in edges]:
        # Assign one-hot encoded edge type based on inversion
        edge_type = edge_type_inverted if weight else edge_type_regular
        g.add_edge(src, tgt, type=edge_type)

    return g

# Create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

'''load and save objects'''


def save_object(obj, filename):
    """Saves a Python object to a compressed file."""
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    """Loads a Python object from a compressed file."""
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def load_module_state(model, state_name):
    """Loads a saved model checkpoint."""
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    return


'''Data preprocessing for AIGER files'''


def calculate_graph_attributes(aig: Aig, G_nx: nx.DiGraph):
    """
    Calculates a vector of 16 structural attributes for a given AIG.
    This vector will be the regression target 'y'.
    """
    # 1. Number of inversions
    num_inversions = sum(1 for po in aig.pos() if po.get_complement()) + \
                     sum(1 for gate in aig.gates() for fanin in aig.fanins(gate) if fanin.get_complement())

    # 2. Number of nodes & 3. Number of edges
    num_nodes = G_nx.number_of_nodes()
    num_edges = G_nx.number_of_edges()

    # 4. Graph depth & 5. Level variance
    levels = np.array([data['level'] for _, data in G_nx.nodes(data=True)])
    depth = np.max(levels) if len(levels) > 0 else 0
    level_variance = np.var(levels) if len(levels) > 1 else 0

    # 6. Avg fanout, 7. Max fanout, 8. Variance fanout
    # Fanouts are now directly extracted from the NetworkX graph attributes.
    fanouts = np.array([data['fanouts'] for _, data in G_nx.nodes(data=True)])
    avg_fanout = np.mean(fanouts) if len(fanouts) > 0 else 0
    max_fanout = np.max(fanouts) if len(fanouts) > 0 else 0
    variance_fanout = np.var(fanouts) if len(fanouts) > 1 else 0

    # 9. Avg edge level span & 10. Variance edge level span
    edge_level_spans = []
    for u, v in G_nx.edges():
        level_u = G_nx.nodes[u].get('level', 0)
        level_v = G_nx.nodes[v].get('level', 0)
        edge_level_spans.append(abs(level_v - level_u))

    avg_edge_level_span = np.mean(edge_level_spans) if edge_level_spans else 0
    variance_edge_level_span = np.var(edge_level_spans) if len(edge_level_spans) > 1 else 0

    # NetworkX-based metrics
    # 11. Density
    density = nx.density(G_nx)

    # 12. Degree assortativity
    assortativity = nx.degree_assortativity_coefficient(G_nx) if num_nodes > 1 else 0

    # Metrics on the undirected version of the largest weakly connected component
    if not G_nx.nodes:
        return np.zeros(16, dtype=np.float32)

    if nx.is_weakly_connected(G_nx):
        G_undirected = G_nx.to_undirected()
    else:
        largest_cc = max(nx.weakly_connected_components(G_nx), key=len)
        G_undirected = G_nx.subgraph(largest_cc).to_undirected()

    # 13. Diameter & 14. Radius & 16. Average Eccentricity
    try:
        if nx.is_connected(G_undirected):
            diameter = nx.diameter(G_undirected)
            radius = nx.radius(G_undirected)
            # FIXED: Calculate average eccentricity manually
            avg_eccentricity = np.mean(list(nx.eccentricity(G_undirected).values()))
        else:
            diameter, radius, avg_eccentricity = -1, -1, -1
    except nx.NetworkXError:
        diameter, radius, avg_eccentricity = -1, -1, -1

    # 15. Algebraic connectivity
    alg_connectivity = nx.algebraic_connectivity(G_undirected) if G_undirected.number_of_nodes() > 1 else 0

    feature_order = [
        'num_edges', 'degree_assortativity', 'num_inversions', 'num_nodes', 'max_fanout',
        'graph_depth', 'var_fanout', 'level_variance', 'avg_edge_level_span',
        'var_edge_level_span', 'diameter', 'radius', 'avg_eccentricity',
        'density', 'algebraic_connectivity', 'avg_fanout'
    ]

    stats = {
        'num_inversions': num_inversions, 'num_nodes': num_nodes, 'num_edges': num_edges,
        'graph_depth': depth, 'level_variance': level_variance, 'avg_fanout': avg_fanout,
        'max_fanout': max_fanout, 'var_fanout': variance_fanout,
        'avg_edge_level_span': avg_edge_level_span,
        'var_edge_level_span': variance_edge_level_span, 'density': density,
        'degree_assortativity': assortativity, 'diameter': diameter, 'radius': radius,
        'algebraic_connectivity': alg_connectivity, 'avg_eccentricity': avg_eccentricity
    }

    return np.array([stats[key] for key in feature_order], dtype=np.float32)


def convert_nx_to_pyg(g_nx: nx.DiGraph):
    """
    Converts a NetworkX graph from aig.to_networkx() to a PyG Data object,
    using the one-hot encoded vectors provided by the library.
    """
    node_map = {node: i for i, node in enumerate(g_nx.nodes())}

    node_features = []
    for node, data in g_nx.nodes(data=True):
        feature_vector = data.get('type')
        if feature_vector is not None:
            node_features.append(torch.tensor(feature_vector, dtype=torch.float))

    edge_list = []
    edge_features = []
    for u, v, data in g_nx.edges(data=True):
        edge_list.append([node_map[u], node_map[v]])
        feature_vector = data.get('type')
        if feature_vector is not None:
            edge_features.append(torch.tensor(feature_vector, dtype=torch.float))

    if not node_features:
        return Data(), {}

    pyg_data = Data(
        x=torch.stack(node_features),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_features) if edge_features else torch.empty(0, 2)
    )
    return pyg_data, node_map


def add_order_info_aig(graph: Data, g_nx: nx.DiGraph, node_map: dict):
    """
    Adds essential DAG-related information to the AIG graph data object.
    - `abs_pe`: The depth-based positional encoding (DAGPE).
    - `dag_rr_edge_index`: The reachability relation for message-passing attention (DAGRA).
    - `mask_rc`: The attention mask for standard attention (DAGRA).
    - Reorders `edge_index` topologically.
    """
    # 1. Calculate node depth for DAGPE using levels from networkx
    # Create a mapping from mapped integer index back to original AIG node object
    idx_to_aig_node = {v: k for k, v in node_map.items()}
    levels = [g_nx.nodes[idx_to_aig_node[i]]['level'] for i in range(graph.num_nodes)]
    graph.abs_pe = torch.tensor(levels, dtype=torch.long)

    # 2. Calculate reachability for DAGRA
    TC = nx.transitive_closure_dag(g_nx)

    # --- FIXED ---
    # Manually create the edge index from the transitive closure graph (TC).
    # The nx.transitive_closure_dag function creates new edges without attributes,
    # which causes an error in from_networkx due to inconsistent attributes.
    # This approach bypasses that check by only extracting the graph structure.
    tc_edge_list = []
    for u, v in TC.edges():
        tc_edge_list.append([node_map[u], node_map[v]])

    if not tc_edge_list:
        tc_edge_index_mapped = torch.empty((2, 0), dtype=torch.long)
    else:
        tc_edge_index_mapped = torch.tensor(tc_edge_list, dtype=torch.long).t().contiguous()

    graph.dag_rr_edge_index = to_undirected(tc_edge_index_mapped)

    # 3. Create the attention mask for DAGRA
    num_nodes = graph.num_nodes
    max_num_nodes = num_nodes
    mask_rc = torch.ones(max_num_nodes, max_num_nodes, dtype=torch.bool)
    for i in range(num_nodes):
        # Find all successors and predecessors in the transitive closure
        successors = tc_edge_index_mapped[1, tc_edge_index_mapped[0] == i]
        predecessors = tc_edge_index_mapped[0, tc_edge_index_mapped[1] == i]

        # Mark self, successors, and predecessors as reachable (False in the mask)
        mask_rc[i, i] = False
        mask_rc[i, successors] = False
        mask_rc[i, predecessors] = False
    graph.mask_rc = mask_rc

    # 4. Topologically sort edges
    topo_order = list(nx.topological_sort(g_nx))
    node_order_map = {node: i for i, node in enumerate(topo_order)}

    # Get original AIG nodes for sorting
    edge_nodes = [(idx_to_aig_node[u.item()], idx_to_aig_node[v.item()]) for u, v in graph.edge_index.t()]

    # Sort edges based on the topological order of their source, then target nodes
    sorted_edge_indices = sorted(
        range(len(edge_nodes)),
        key=lambda i: (node_order_map[edge_nodes[i][0]], node_order_map[edge_nodes[i][1]])
    )

    graph.edge_index_original = graph.edge_index.clone()
    graph.edge_index = graph.edge_index[:, sorted_edge_indices]
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        graph.edge_attr = graph.edge_attr[sorted_edge_indices]

    return graph


def process_and_save_aig_graphs(data_dir, output_path):
    """
    Loads all .aig files from a directory, converts them to PyG graphs with
    ordering info, calculates structural attributes, and saves the entire
    dataset to a single file.
    """
    g_list = []
    max_n = 0

    try:
        aiger_files = [f for f in os.listdir(data_dir) if f.endswith('.aig')]
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' was not found.")
        return

    if not aiger_files:
        print(f"No .aig files found in '{data_dir}'.")
        return

    for filename in tqdm(aiger_files, desc="Loading and processing AIGER files"):
        filepath = os.path.join(data_dir, filename)
        try:
            aig = aigverse.read_aiger_into_aig(filepath)
            # OPTIMIZED: Now creating the NetworkX graph with fanouts included
            g_nx = to_networkx(aig, levels=True, fanouts=True)
            g_pyg, node_map = convert_nx_to_pyg(g_nx)

            if g_pyg.num_nodes == 0:
                print(f"Skipping empty or invalid graph: {filename}")
                continue

            # Add ordering information
            g_pyg = add_order_info_aig(g_pyg, g_nx, node_map)

            # This function now uses the fanouts from g_nx
            attributes = calculate_graph_attributes(aig, g_nx)

            max_n = max(max_n, g_pyg.num_nodes)
            g_list.append((g_pyg, attributes))
        except Exception as e:
            print(f"Could not process file {filename}: {e}")

    ng = len(g_list)
    if ng == 0:
        print("No graphs were successfully loaded. No file will be saved.")
        return

    print(f'\nSuccessfully processed {ng} graphs.')
    print(f'Maximum # nodes: {max_n}')

    # Save the entire list to one file
    print(f"Saving processed graphs to {output_path}...")
    save_object(g_list, output_path)
    print("Processing and saving complete.")


if __name__ == '__main__':
    # Example of how to run the script from the command line
    parser = argparse.ArgumentParser(description='Preprocess AIGER files into a PyG data file.')
    parser.add_argument('--data-dir', type=str, default="../trial_data", help='Directory containing the .aig files.')
    parser.add_argument('--output-path', type=str, default='../processed_data/processed_data.pkl.gz', help='Path to save the output .pkl.gz file.')
    args = parser.parse_args()

    process_and_save_aig_graphs(args.data_dir, args.output_path)
