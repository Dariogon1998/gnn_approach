'''
filename: first.py
Author: Dario Pullia
Date created: 04/10/2023

Description:


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
from torch_geometric.data import Dataset

sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img
# Custom machine learning libraries
import dataset as myds
import model as mymodel


print(f"PyTorch Version: {torch.__version__}")
print(f"Torch Geometric Version: {torch_geometric.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

filename = '/eos/user/d/dapullia/tp_dataset/snana_hits.txt'

# Create the dataset

train_dataset = myds.groupsDataset(root='/afs/cern.ch/work/d/dapullia/public/dune/gnn_approach/data/', filename=filename, test=False, n_tps_to_read=10000)
test_dataset = myds.groupsDataset(root='/afs/cern.ch/work/d/dapullia/public/dune/gnn_approach/data/', filename=filename, test=True)

# See the dataset

print(train_dataset)
print(test_dataset)


# # Create the model

# model_params = {
#     "model_embedding_size": 64,
#     "model_attention_heads": 3,
#     "model_layers": 4,
#     "model_dropout_rate": 0.2,
#     "model_top_k_ratio": 0.5,
#     "model_top_k_every_n": 1,
#     "model_dense_neurons": 256,
# }

# gnn_model = mymodel.GNN(feature_size=7, model_params=model_params)

# # Train the model




# # Test the model


