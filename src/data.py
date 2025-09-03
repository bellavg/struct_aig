# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
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
    """
    def __init__(self, processed_file_path):
        """
        Args:
            processed_file_path (str): The path to the .pkl.gz file containing
                                       the dictionary of data and normalization stats.
        """
        loaded_data = load_object(processed_file_path)
        self.data_list = loaded_data['data']
        self.mean = loaded_data.get('mean')
        self.std = loaded_data.get('std')

    def __len__(self):
        """Returns the total number of graphs in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves the graph and its corresponding attributes at a given index,
        and assigns them to the graph's 'y' and 'y_raw' properties.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            torch_geometric.data.Data: A graph data object ready for the model.
        """
        graph, (raw_attributes, std_attributes) = self.data_list[idx]

        graph.y = std_attributes
        graph.y_raw = raw_attributes

        return graph

def collate_aig(data_list):
    """
    A custom collate function for the PyTorch DataLoader.
    It clones the data to avoid in-place modification and handles the
    variable-sized masks.
    """
    # --- CHANGE: Clone the data objects before modifying them ---
    cloned_data_list = [d.clone() for d in data_list]

    # 1. Separate the variable-sized masks from the cloned data objects.
    masks = [data.mask_rc for data in cloned_data_list]
    for data in cloned_data_list:
        del data.mask_rc

    # 2. Create a standard PyG batch from the modified clones.
    batch = Batch.from_data_list(cloned_data_list)

    # 3. Attach the list of original, unpadded masks to the batch object.
    batch.mask_rc_list = masks

    return batch