'''
filename: first.py
Author: Dario Pullia
Date created: 04/10/2023

Description:


'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import argparse
import warnings
import sys

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader   


sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img
# Custom machine learning libraries
import dataset as myds
import model as mymodel
import train as mytrain

# set the random seed
torch.manual_seed(42)
np.random.seed(42)



N_TPS_TO_READ = 5000000
N_EPOCHS_TO_TRAIN = 400
N_EPOCHS_TO_TEST = 5


def get_unique_from_loader(loader, return_counts=False):
    '''
    This function returns the unique labels in the dataset.
    '''
    lab_list = []
    for batch in loader:
        lab_list.append(np.argmax(batch.y.cpu().detach().numpy()))
    
    lab_list = np.array(lab_list)
    return np.unique(lab_list, return_counts=return_counts)



print(f"PyTorch Version: {torch.__version__}")
print(f"Torch Geometric Version: {torch_geometric.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

filename = '/eos/user/d/dapullia/tp_dataset/snana_hits.txt'

# Create the dataset

dataset = myds.groupsDataset(root='/afs/cern.ch/work/d/dapullia/public/dune/gnn_approach/data/', filename=filename, test=False, n_tps_to_read=N_TPS_TO_READ, balance_classes=True)

# shuffle the dataset
dataset = dataset.shuffle()

# split the dataset
train_test_split = int(len(dataset)*0.8)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_test_split, len(dataset) - train_test_split])


print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

print(get_unique_from_loader(train_dataset, return_counts=True))
print(get_unique_from_loader(test_dataset, return_counts=True))

# Create the model

model_params = {
    "model_embedding_size": 16,
    "model_attention_heads": 2,
    "model_layers": 1,
    "model_dropout_rate": 0.2,
    "model_top_k_ratio": 0.5,
    "model_top_k_every_n": 100,
    "model_dense_neurons": 128,
    "model_first_layer_neurons": 32,
}

print("Creating the model")
gnn_model = mymodel.GNN(feature_size=6, model_params=model_params)
print("Model created")
# Train the model

# Prepare training  
print("Preparing training")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
print("Training prepared")
# Loading the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
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
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
best_epochs_list = []
for epoch in range(N_EPOCHS_TO_TRAIN): 
    if early_stopping_counter <= 10: # = x * 5 
        # Training
        model.train()
        loss, accuracy = mytrain.train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch} | Train Loss {loss} | Train Accuracy {accuracy}")
        train_losses.append(loss)
        train_accuracies.append(accuracy)
        # Testing
        model.eval()
        if epoch % N_EPOCHS_TO_TEST == 0:
            loss, accuracy = mytrain.test(epoch, model, test_loader, loss_fn)
            test_losses.append(loss)
            test_accuracies.append(accuracy)

            print("|-------------------------------------------------------------------------|")
            print(f"Epoch {epoch} | Test Loss {loss} | Test Accuracy {accuracy}")
            print("|-------------------------------------------------------------------------|")            
            # Update best loss
            if float(loss) < best_loss:
                best_loss = loss
                # Save the currently best model 
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')
                early_stopping_counter = 0
                best_epoch = [epoch, loss, accuracy]
                print("New best model! Epoch: ", best_epoch)
                best_epochs_list.append(best_epoch)
            else:
                early_stopping_counter += 1
        scheduler.step()
    else:
        print("Early stopping")
        print(f"Best epochs: {best_epochs_list}")
        break


# Plot the loss
plt.plot(np.arange(len(train_losses)), train_losses, label="Train loss")
plt.plot(np.arange(len(test_losses))*N_EPOCHS_TO_TEST, test_losses, label="Test loss")
plt.xlabel("Epoch")
plt.title("Loss")
plt.savefig("loss.png")
plt.clf()

# Plot the accuracy
plt.plot(np.arange(len(train_accuracies)), train_accuracies, label="Train accuracy")
plt.plot(np.arange(len(test_accuracies))*N_EPOCHS_TO_TEST, test_accuracies, label="Test accuracy")
plt.xlabel("Epoch")
plt.title("Accuracy")
plt.savefig("accuracy.png")
plt.clf()

# Calculate metrics on the test set
all_preds = []
all_labels = []

for batch in test_loader:
    batch.to(device)  
    pred = model(batch.x.float(), 
                    batch.edge_index, 
                    batch.batch) 

    batch.y=torch.reshape(batch.y, pred.shape)

    all_preds.append((F.softmax(pred).cpu().detach().numpy() ))
    all_labels.append((batch.y.cpu().detach().numpy()))

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)



cm, accuracy, precision, recall, f1 = mytrain.calculate_metrics(all_labels, all_preds)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Confusion matrix: \n{cm}")
# Plot the confusion matrix
labels = [0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(10,10))
plt.title("Confusion matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
plt.clf()


# Calculate metrics on the train set
all_preds = []
all_labels = []

for batch in train_loader:
    batch.to(device)  
    pred = model(batch.x.float(), 
                    batch.edge_index, 
                    batch.batch) 

    batch.y=torch.reshape(batch.y, pred.shape)

    all_preds.append((F.softmax(pred).cpu().detach().numpy() ))
    all_labels.append((batch.y.cpu().detach().numpy()))

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)



cm, accuracy, precision, recall, f1 = mytrain.calculate_metrics(all_labels, all_preds)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Confusion matrix: \n{cm}")
# Plot the confusion matrix
labels = [0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(10,10))
plt.title("Confusion matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_train.png')
plt.clf()