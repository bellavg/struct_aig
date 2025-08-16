# -*- coding: utf-8 -*-
from __future__ import print_function
import gzip
import pickle
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
import os
import argparse
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, from_networkx

# --- NEW: Import aigverse and its views ---
import aigverse
from aigverse import Aig

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
    num_inversions = sum(1 for po in aig.pos if po.is_complemented) + \
                     sum(1 for gate in aig.gates for fanin in gate.fanins if fanin.is_complemented)

    # 2. Number of nodes & 3. Number of edges
    num_nodes = G_nx.number_of_nodes()
    num_edges = G_nx.number_of_edges()

    # 4. Graph depth & 5. Level variance
    levels = np.array([data['level'] for _, data in G_nx.nodes(data=True)])
    depth = np.max(levels) if len(levels) > 0 else 0
    level_variance = np.var(levels) if len(levels) > 1 else 0

    # 6. Avg fanout, 7. Max fanout, 8. Variance fanout
    fanouts = np.array([aig.fanout_size(n) for n in aig.nodes])
    avg_fanout = np.mean(fanouts) if len(fanouts) > 0 else 0
    max_fanout = np.max(fanouts) if len(fanouts) > 0 else 0
    variance_fanout = np.var(fanouts) if len(fanouts) > 1 else 0

    # 9. Avg edge level span & 10. Variance edge level span
    edge_level_spans = []
    for u, v, data in G_nx.edges(data=True):
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
            avg_eccentricity = nx.average_eccentricity(G_undirected)
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
    tc_data = from_networkx(TC)
    # We need to map the nodes in tc_data back to our pyg graph's indices
    tc_edge_index_mapped = torch.empty_like(tc_data.edge_index)
    for i, edge in enumerate(tc_data.edge_index.t()):
        u, v = edge[0].item(), edge[1].item()
        # tc_data.n_id are the original node objects from g_nx
        tc_edge_index_mapped[0, i] = node_map[tc_data.n_id[u]]
        tc_edge_index_mapped[1, i] = node_map[tc_data.n_id[v]]

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
            g_nx = aig.to_networkx(levels=True, po_level_plus_one=True, one_hot_types=True)
            g_pyg, node_map = convert_nx_to_pyg(g_nx)

            if g_pyg.num_nodes == 0:
                print(f"Skipping empty or invalid graph: {filename}")
                continue

            # Add ordering information
            g_pyg = add_order_info_aig(g_pyg, g_nx, node_map)

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
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the .aig files.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output .pkl.gz file.')
    args = parser.parse_args()

    process_and_save_aig_graphs(args.data_dir, args.output_path)
