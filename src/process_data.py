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
import pandas as pd
# --- NEW: Import aigverse and its views ---
import aigverse
from typing import Any, Final

import networkx as nx
import numpy as np

from aigverse import Aig, DepthAig, simulate, simulate_nodes, to_edge_list
import aigverse.adapters

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
    mask_rc = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
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


def process_and_save_aig_graphs(data_dir, output_path, max_nodes=50000):
    """
    Loads all .aig files from subdirectories, processes them, loads pre-calculated wirelengths,
    sorts by graph size, computes normalization stats, and saves everything to a single file.
    """
    g_list = []
    attributes_list = []  # This will store the raw wirelengths
    max_n = 0

    # Walk through the directory structure to find subfolders with wirelength files
    for subdir, _, files in os.walk(data_dir):
        if 'wirelength' in files:
            wirelength_file_path = os.path.join(subdir, 'wirelength')

            # Read wirelengths into a dictionary for easy lookup
            wirelength_df = pd.read_csv(wirelength_file_path, delim_whitespace=True, header=None,
                                        names=['aig_file', 'wirelength'])
            wirelength_dict = wirelength_df.set_index('aig_file')['wirelength'].to_dict()

            aiger_files = [f for f in files if f.endswith('.aig')]

            if not aiger_files:
                print(f"No .aig files found in '{subdir}'.")
                continue

            for filename in tqdm(aiger_files, desc=f"Processing AIGER files in {subdir}"):
                if filename in wirelength_dict:
                    filepath = os.path.join(subdir, filename)
                    try:
                        aig = aigverse.read_aiger_into_aig(filepath)
                        g_nx = aig.to_networkx(levels=True, fanouts=True)
                        g_pyg, node_map = convert_nx_to_pyg(g_nx)

                        # --- ADDED: Filter graphs by number of nodes ---
                        if g_pyg.num_nodes > max_nodes:
                            print(f"Skipping graph {filename} with {g_pyg.num_nodes} nodes (limit is {max_nodes}).")
                            continue

                        if g_pyg.num_nodes == 0:
                            print(f"Skipping empty or invalid graph: {filename}")
                            continue

                        g_pyg = add_order_info_aig(g_pyg, g_nx, node_map)

                        # Get the pre-calculated wirelength from the dictionary
                        wirelength = wirelength_dict[filename]
                        attributes = np.array([wirelength], dtype=np.float32)

                        max_n = max(max_n, g_pyg.num_nodes)

                        g_list.append(g_pyg)
                        attributes_list.append(attributes)
                    except Exception as e:
                        print(f"Could not process file {filename}: {e}")
                else:
                    print(f"Wirelength not found for {filename} in {wirelength_file_path}")

    ng = len(g_list)
    if ng == 0:
        print("No graphs were successfully loaded. No file will be saved.")
        return

    print(f'\nSuccessfully processed {ng} graphs.')
    print(f'Maximum # nodes: {max_n}')

    # --- Sort graphs and attributes by the number of nodes ---
    print("Sorting graphs by number of nodes...")
    combined = list(zip(g_list, attributes_list))
    combined.sort(key=lambda x: x[0].num_nodes)
    g_list_sorted, attributes_list_sorted = zip(*combined)

    # --- Standardize the wirelength values ---
    print("Standardizing wirelength values...")
    attributes_tensor = torch.tensor(np.array(attributes_list_sorted), dtype=torch.float)
    mean = attributes_tensor.mean(dim=0)
    std = attributes_tensor.std(dim=0)
    std[std == 0] = 1e-8

    standardized_attributes = (attributes_tensor - mean) / std

    # --- Combine graphs with both raw and standardized attributes ---
    final_data_list = []
    for i in range(len(g_list_sorted)):
        raw_attr = torch.tensor(attributes_list_sorted[i], dtype=torch.float)
        std_attr = standardized_attributes[i]
        final_data_list.append((g_list_sorted[i], (raw_attr, std_attr)))

    # Create a dictionary to save everything
    save_data = {
        'data': final_data_list,
        'mean': mean,
        'std': std,
    }

    # Save the dictionary object
    print(f"Saving processed data and stats to {output_path}...")
    save_object(save_data, output_path)
    print("Processing and saving complete.")


if __name__ == '__main__':
    # Example of how to run the script from the command line
    parser = argparse.ArgumentParser(description='Preprocess AIGER files into a PyG data file.')
    parser.add_argument('--data-dir', type=str, default="../trial_data", help='Directory containing the .aig files.')
    parser.add_argument('--output-path', type=str, default='../processed_data/processed_data.pkl.gz',
                        help='Path to save the output .pkl.gz file.')
    parser.add_argument('--max-nodes', type=int, default=50000,
                        help='Maximum number of nodes to include in the dataset.')
    args = parser.parse_args()

    process_and_save_aig_graphs(args.data_dir, args.output_path, args.max_nodes)
