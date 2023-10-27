'''
filename: gnn_classifier.py
Author: Dario Pullia
Date created: 17/10/2023

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

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", help="input data file", default="/eos/user/d/dapullia/tp_dataset/snana_hits.txt", type=str)
parser.add_argument("--output_folder", help="save path", default="/eos/user/d/dapullia/tp_dataset/", type=str)
parser.add_argument("--dataset_folder", help="folder to save the dataset", default="/afs/cern.ch/work/d/dapullia/public/dune/gnn_approach/data/", type=str)
parser.add_argument("--model_name", help="model name", default="model.h5", type=str)
parser.add_argument('--load_model', action='store_true', help='save the model')
parser.add_argument("--balance_training_set", action='store_true', help="balance the training set")
parser.add_argument("--n_tps_to_read", help="number of TPs to read", default=1000000, type=int)
parser.add_argument("--n_epochs_to_train", help="number of epochs to train", default=400, type=int)
parser.add_argument("--n_epochs_to_test", help="number of epochs to test", default=5, type=int)
parser.add_argument("--batch_size", help="batch size", default=32, type=int)
parser.add_argument("--train_test_split", help="train/test split", default=0.8, type=float)
parser.add_argument("--early_stopping_patience", help="early stopping", default=10, type=int)

args = parser.parse_args()
input_data = args.input_data
output_folder = args.output_folder
dataset_folder = args.dataset_folder
model_name = args.model_name
load_model = args.load_model
balance_training_set = args.balance_training_set
n_tps_to_read = args.n_tps_to_read
n_epochs_to_train = args.n_epochs_to_train
n_epochs_to_test = args.n_epochs_to_test
batch_size = args.batch_size
train_test_split = args.train_test_split
early_stopping_patience = args.early_stopping_patience

output_folder = output_folder + model_name + "/"

if __name__=='__main__':
    if not os.path.exists(input_data):
        print(input_data)
        print("Exists input data: ", os.path.exists(input_data))
        print("Input file not found.")
        exit()

    # create the output folders
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+"checkpoints/"):
        os.makedirs(output_folder+"checkpoints/")
    if not os.path.exists(output_folder+"log/test/"):
        os.makedirs(output_folder+"log/test/")
    if not os.path.exists(output_folder+"log/train/"):
        os.makedirs(output_folder+"log/train/")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torch Geometric Version: {torch_geometric.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Create the dataset
    print("Creating the dataset...")
    dataset = myds.groupsDataset(root=dataset_folder, filename=input_data, n_tps_to_read=n_tps_to_read, balance_training_set=balance_training_set, test = False)
    # dataset.shuffle()
    print("Dataset created.")
    
    # split the dataset
    print("Splitting the dataset...")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*train_test_split), len(dataset) - int(len(dataset)*train_test_split)])
    print(f"Number of training graphs: {len(train_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")

    # print(myds.get_unique_from_loader(train_dataset, return_counts=True))
    # print(myds.get_unique_from_loader(test_dataset, return_counts=True))
    print("Dataset splitted.")
    # Create the model
    model_params = {
    "model_embedding_size": 32,
    "model_attention_heads": 2,
    "model_layers": 2,
    "model_dropout_rate": 0.2,
    "model_top_k_every_n": 100,
    "model_dense_neurons": 128,
    "model_first_layer_neurons": 32,
    }
    print("Creating the model")
    gnn_model = mymodel.GNN(feature_size=6, model_params=model_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = gnn_model.to(device)
    print(f"Number of parameters: {mytrain.count_parameters(model)}")

    # Create weights for the loss function to account for the unbalanced dataset
    print("Creating weights for the loss function to account for the unbalanced dataset")
    unique, counts = myds.get_unique_from_loader(train_dataset, return_counts=True)
    class_weights = torch.FloatTensor(1/counts).to(device)
    print("Weights created")

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print("Model created")

    # Prepare training  
    print("Preparing training")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    print("Training prepared")


    # Train the model
    print("Training the model...")

    # Start training
    best_loss = 1000
    early_stopping_counter = 0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_epochs_list = []

    for epoch in range(n_epochs_to_train): 
        if early_stopping_counter <= early_stopping_patience:
            # Training
            model.train()
            loss, accuracy = mytrain.train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, output_folder)
            print(f"Epoch {epoch} | Train Loss {loss} | Train Accuracy {accuracy}")
            train_losses.append(loss)
            train_accuracies.append(accuracy)
            # Testing
            model.eval()
            if epoch % n_epochs_to_test == 0:
                loss, accuracy = mytrain.test(epoch, model, test_loader, loss_fn, output_folder)
                test_losses.append(loss)
                test_accuracies.append(accuracy)

                print("|-------------------------------------------------------------------------|")
                print(f"Epoch {epoch} | Test Loss {loss} | Test Accuracy {accuracy}")
                print("|-------------------------------------------------------------------------|")            
                # Update best loss
                if float(loss) < best_loss:
                    best_loss = loss
                    # Save the currently best model 
                    torch.save(model.state_dict(), output_folder+"checkpoints/"+model_name)
                    early_stopping_counter = 0
                    best_epoch = [epoch, loss, accuracy]
                    print("New best model! Epoch: ", best_epoch)
                    best_epochs_list.append(best_epoch)
                else:
                    early_stopping_counter += 1
            scheduler.step()
        else:
            print("Early stopping")
            print(f"Best epochs: {best_epochs_list}\n\n")
            break

    # # Plot the loss
    # plt.plot(np.arange(len(train_losses)), train_losses, label="Train loss")
    # plt.plot(np.arange(len(test_losses))*n_epochs_to_train, test_losses, label="Test loss")
    # plt.xlabel("Epoch")
    # plt.title("Loss")
    # plt.savefig(output_folder+"loss.png")
    # plt.clf()

    # # Plot the accuracy
    # plt.plot(np.arange(len(train_accuracies)), train_accuracies, label="Train accuracy")
    # plt.plot(np.arange(len(test_accuracies))*n_epochs_to_train, test_accuracies, label="Test accuracy")
    # plt.xlabel("Epoch")
    # plt.title("Accuracy")
    # plt.savefig(output_folder+"accuracy.png")
    # plt.clf()

    # # Calculate metrics on the test set
    # all_preds = []
    # all_labels = []

    # for batch in test_loader:
    #     batch.to(device)  
    #     pred = model(batch.x.float(), 
    #                     batch.edge_index, 
    #                     batch.batch) 

    #     batch.y=torch.reshape(batch.y, pred.shape)

    #     all_preds.append((F.softmax(pred).cpu().detach().numpy() ))
    #     all_labels.append((batch.y.cpu().detach().numpy()))

    # all_preds = np.concatenate(all_preds)
    # all_labels = np.concatenate(all_labels)


    # mytrain.test(999, model, test_loader, loss_fn, output_folder)

    # cm, accuracy, precision, recall, f1 = mytrain.calculate_metrics(all_labels, all_preds)

    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1: {f1}")
    # print(f"Confusion matrix: \n{cm}")
    # # Plot the confusion matrix
    # labels = [0,1,2,3,4,5,6,7,8,9]
    # plt.figure(figsize=(10,10))
    # plt.title("Confusion matrix")
    # sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.savefig(output_folder+'confusion_matrix.png')
    # plt.clf()


    # # Calculate metrics on the train set
    # all_preds = []
    # all_labels = []

    # for batch in train_loader:
    #     batch.to(device)  
    #     pred = model(batch.x.float(), 
    #                     batch.edge_index, 
    #                     batch.batch) 

    #     batch.y=torch.reshape(batch.y, pred.shape)

    #     all_preds.append((F.softmax(pred).cpu().detach().numpy() ))
    #     all_labels.append((batch.y.cpu().detach().numpy()))

    # all_preds = np.concatenate(all_preds)
    # all_labels = np.concatenate(all_labels)


    # mytrain.test(999, model, train_loader, loss_fn, output_folder)

    # cm, accuracy, precision, recall, f1 = mytrain.calculate_metrics(all_labels, all_preds)

    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1: {f1}")
    # print(f"Confusion matrix: \n{cm}")
    # # Plot the confusion matrix
    # labels = [0,1,2,3,4,5,6,7,8,9]
    # plt.figure(figsize=(10,10))
    # plt.title("Confusion matrix")
    # sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.savefig(output_folder+'confusion_matrix_train.png')
    # plt.clf()





