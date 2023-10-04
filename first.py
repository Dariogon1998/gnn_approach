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
from torch_geometric.data import Dataset, Data, DataLoader

sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img
# Custom machine learning libraries
import dataset as myds
import model as mymodel
import train as mytrain


print(f"PyTorch Version: {torch.__version__}")
print(f"Torch Geometric Version: {torch_geometric.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

filename = '/eos/user/d/dapullia/tp_dataset/snana_hits.txt'

# Create the dataset

dataset = myds.groupsDataset(root='/afs/cern.ch/work/d/dapullia/public/dune/gnn_approach/data/', filename=filename, test=False, n_tps_to_read=1000)

# shuffle the dataset
dataset = dataset.shuffle()

# split the dataset
train_test_split = int(len(dataset)*0.8)

train_dataset = dataset[:train_test_split]
test_dataset = dataset[train_test_split:]


print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

print(train_dataset) 


# Create the model

model_params = {
    "model_embedding_size": 64,
    "model_attention_heads": 3,
    "model_layers": 4,
    "model_dropout_rate": 0.2,
    "model_top_k_ratio": 0.5,
    "model_top_k_every_n": 1,
    "model_dense_neurons": 256,
}

gnn_model = mymodel.GNN(feature_size=6, model_params=model_params)

# Train the model

# Prepare training  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Loading the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gnn_model.to(device)
print(f"Number of parameters: {mytrain.count_parameters(model)}")

# < 1 increases precision, > 1 recall   
weight = torch.tensor([1], dtype=torch.float32).to(device)
# loss as categorical cross entropy
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Start training
best_loss = 1000
early_stopping_counter = 0
for epoch in range(10): 
    if early_stopping_counter <= 10: # = x * 5 
        # Training
        model.train()
        loss = mytrain.train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch} | Train Loss {loss}")

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = mytrain.test(epoch, model, test_loader, loss_fn)
            print(f"Epoch {epoch} | Test Loss {loss}")
            
            # Update best loss
            if float(loss) < best_loss:
                best_loss = loss
                # Save the currently best model 
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
        scheduler.step()
    else:
        print("Early stopping")
        break




# Test the model


