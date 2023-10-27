'''
filename: dataset.py
Author: Dario Pullia
Date created: 04/10/2023

Description:
This file contains the class that defines the dataset for the GNN approach.

The raw data is a txt file with all the TPs in the dataset.
Each group of TPs is a graph with TPs as nodes and edges between TPs that have still to be defined.
Each node has the TP variables as features.

'''


import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import warnings
import sys

import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.data import Dataset, Data

sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img
# set the random seed
torch.manual_seed(42)
np.random.seed(42)



class groupsDataset(Dataset):

    def __init__(self, root, filename, n_tps_to_read=1000, test=False, transform=None, pre_transform=None, balance_training_set=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.n_tps_to_read = n_tps_to_read
        self.balance_training_set = balance_training_set
        super(groupsDataset, self).__init__(root, transform, pre_transform)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        data = np.loadtxt(self.raw_paths[0], max_rows=self.n_tps_to_read, dtype=int)
        channel_map = tp2img.create_channel_map_array('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/channel-maps/channel_map_upright.txt')
        # groups = tp2img.group_maker_only_by_time(data, channel_map, ticks_limit=200, channel_limit=20, min_tps_to_group=3)
        groups = tp2img.group_maker(data, channel_map, ticks_limit=200, channel_limit=20, min_tps_to_group=2)

        # remove the groups with different types
        groups = [group for group in groups if len(set(group[:, 7])) == 1]
        if self.balance_training_set:
            # balance the classes
            groups = self._balance_training_set(groups)
        self.groups = groups

        if self.test:
            return [f'data_test_{i}.pt' for i in range(len(groups)) ]
        else:
            return [f'data_{i}.pt' for i in range(len(groups))]

    def download(self):
        pass

    def process(self):
        data = np.loadtxt(self.raw_paths[0], max_rows=self.n_tps_to_read, dtype=int)
        channel_map = tp2img.create_channel_map_array('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/channel-maps/channel_map_upright.txt')
        # groups = tp2img.group_maker_only_by_time(data, channel_map, ticks_limit=200, channel_limit=20, min_tps_to_group=3)
        groups = tp2img.group_maker(data, channel_map, ticks_limit=150, channel_limit=20, min_tps_to_group=2)

        # remove the groups with different types
        groups = [group for group in groups if len(set(group[:, 7])) == 1]
        if self.balance_training_set:
            # balance the classes
            groups = self._balance_training_set(groups)

        self.groups = groups

        # create the graphs
        for i, group in enumerate(groups):
            group = torch.tensor(group, dtype=torch.int)
            node_features = self._get_node_features(group)
            edge_index = self._get_edge_index(group)
            label = self._get_label(group)
            data = Data(x=node_features, edge_index=edge_index, y=label)

            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{i}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
    
    def _get_node_features(self, group):
        '''
        This function returns the node features for a group of TPs.
        Shape: [n_nodes, n_node_features]
        '''
        return group[:, :6]

    def _get_edge_index(self, group):
        '''
        This function returns the edge index for a group of TPs.
        At first, we connect all the nodes to all the other nodes.
        Shape: [2, n_edges]
        '''
        # create the adjacency matrix
        # [[1,1,0,0,0],]
        # [[1,1,1,0,0],]
        # [[0,1,1,1,0],]
        # [[0,0,1,1,1],]
        # [[0,0,0,1,1],]
        adjacency_matrix = np.zeros((len(group), len(group)))
        adjacency_matrix[0, 0] = 1
        adjacency_matrix[0, 1] = 1
        for i in range(1,len(group)-1):
            adjacency_matrix[i, i] = 1
            adjacency_matrix[i, i-1] = 1
            adjacency_matrix[i, i+1] = 1
        adjacency_matrix[-1, -1] = 1
        adjacency_matrix[-1, -2] = 1
        
        # turn the adjacency matrix into a list of edges
        edge_index = np.array(np.where(adjacency_matrix == 1))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return edge_index

    def _get_label(self, group):
        '''
        This function returns the label for a group of TPs. 
        '''
        # return the type of the group in one-hot encoding
        label = np.zeros(10)
        label[group[0, 7]] = 1
        label = torch.tensor(label, dtype=torch.uint8)
        return label

    def _balance_training_set(self, groups):
        '''
        This function balances the classes in the dataset (not definitive)
        '''
        # randomly remove 80% of the groups of type 1
        n_groups =  []
        for group in groups:
            if group[0, 7] == 1:
                if np.random.rand() < 0.15:
                    n_groups.append(group)
            else:
                n_groups.append(group)
        return n_groups

    def len(self):
        return len(self.groups)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data


def get_unique_from_loader(loader, return_counts=False):
    '''
    This function returns the unique labels in the dataset.
    '''
    lab_list = []
    for batch in loader:
        lab_list.append(np.argmax(batch.y.cpu().detach().numpy()))
    
    lab_list = np.array(lab_list)
    return np.unique(lab_list, return_counts=return_counts)








