# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pickle
import gzip

def load_object(filename):
    """
    Helper function to load a Python object from a compressed gzip file.
    This is used to load the dataset pre-processed by data_utils.py.
    """
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)

class AIGDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed AIG graphs and their
    structural attributes for a regression task.

    This class is designed to work with the output of the
    `process_and_save_aig_graphs` function from `data_utils.py`.
    """
    def __init__(self, processed_file_path):
        """
        Args:
            processed_file_path (str): The path to the .pkl.gz file containing
                                       the list of (graph, attributes) tuples.
        """
        # Load the entire dataset into memory. The dataset is a list where
        # each item is a tuple: (PyG_Data_object, numpy_array_of_attributes)
        self.data_list = load_object(processed_file_path)

    def __len__(self):
        """Returns the total number of graphs in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves the graph and its corresponding attributes at a given index,
        assigning the attributes to the graph's 'y' property for the model.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            torch_geometric.data.Data: A graph data object ready for the model,
                                       with the target vector in `data.y`.
        """
        # Retrieve the pre-processed graph and its structural attributes
        graph, attributes = self.data_list[idx]

        # The GraphTransformer model expects the regression target to be in the 'y'
        # attribute of the data object. We convert the attributes to a tensor.
        graph.y = torch.tensor(attributes, dtype=torch.float)

        return graph

def collate_aig(data_list):
    """
    A custom collate function for the PyTorch DataLoader.

    It pads the `mask_rc` tensor, which is 2D and has a variable size
    ([num_nodes, num_nodes]), to the maximum number of nodes in the batch.
    This prevents errors during the batching process. Other attributes are
    handled correctly by the default `Batch.from_data_list` method.
    """
    # Find the maximum number of nodes in the current batch
    max_nodes = max([data.num_nodes for data in data_list])

    # Pad only the `mask_rc` tensor for each graph in the batch
    for data in data_list:
        num_nodes = data.num_nodes

        # The original mask is (num_nodes, num_nodes). We pad it to (max_nodes, max_nodes).
        # The padding value of True means that the padded positions are masked (ignored).
        padded_mask = torch.ones((max_nodes, max_nodes), dtype=torch.bool)
        padded_mask[:num_nodes, :num_nodes] = data.mask_rc
        data.mask_rc = padded_mask

    # Now that the problematic 2D tensor is padded, we can safely create the batch
    return Batch.from_data_list(data_list)