#%% imports 
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset import groupsDataset
from model import GNN
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.batch) 
        # Calculating the loss and gradients

        batch.y = torch.reshape(batch.y, pred.shape)

        loss = loss_fn(pred, batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_index, 
                        batch.batch) 
        loss = loss_fn(pred, torch.reshape(batch.y, pred.shape).float())

         # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    print("implement a function to log the confusion matrix")
    # cm = confusion_matrix(y_pred, y_true)
    # classes = ["0", "1"]
    # df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    # plt.figure(figsize = (10,7))
    # cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    # cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')
    # mlflow.log_artifact(f"data/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true, epoch, type):
    print("implement a function to calculate metrics")
#     print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
#     print(f"F1 Score: {f1_score(y_true, y_pred)}")
#     print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
#     prec = precision_score(y_true, y_pred)
#     rec = recall_score(y_true, y_pred)
#     print(f"Precision: {prec}")
#     print(f"Recall: {rec}")
#     try:
#         roc = roc_auc_score(y_true, y_pred)
#         print(f"ROC AUC: {roc}")
#     except:
#         print(f"ROC AUC: notdefined")



