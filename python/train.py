#%% imports 
import torch 
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset import groupsDataset
from model import GNN
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set the random seed
torch.manual_seed(42)
np.random.seed(42)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, output_folder="log/"):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_accuracy = 0.0
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
        # all_preds.append((F.softmax(pred).cpu().detach().numpy() ))
    
        running_accuracy += accuracy_score(np.argmax(batch.y.cpu().detach().numpy(), axis=1), np.argmax((pred).cpu().detach().numpy(), axis=1))
        all_preds.append(F.softmax(pred, dim=1).cpu().detach().numpy())
        all_labels.append((batch.y.cpu().detach().numpy()))


    log_metrics(np.concatenate(all_labels), np.concatenate(all_preds), epoch=epoch, test=False, output_folder=output_folder)

    return running_loss/step, running_accuracy/step


def test(epoch, model, test_loader, loss_fn, output_folder="log/"):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    running_accuracy = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_index, 
                        batch.batch) 

        batch.y=torch.reshape(batch.y, pred.shape)
        loss = loss_fn(pred, batch.y.float())

         # Update tracking
        running_loss += loss.item()
        step += 1
        # all_preds.append((F.softmax(pred).cpu().detach().numpy() ))
        all_preds.append(F.softmax(pred, dim=1).cpu().detach().numpy())
        all_labels.append((batch.y.cpu().detach().numpy()))

        running_accuracy += accuracy_score(np.argmax(batch.y.cpu().detach().numpy(), axis=1), np.argmax((pred).cpu().detach().numpy(), axis=1))

    log_metrics(np.concatenate(all_labels), np.concatenate(all_preds), epoch=epoch, test=True, output_folder=output_folder)

    return running_loss/step, running_accuracy/step

def calculate_metrics( y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9]):
    # calculate the confusion matrix, the accuracy, and the precision and recall 
    y_pred_am = np.argmax(y_pred, axis=1)
    y_true_am = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true_am, y_pred_am, labels=labels)
    accuracy = accuracy_score(y_true_am, y_pred_am)
    precision = precision_score(y_true_am, y_pred_am, average='macro')
    recall = recall_score(y_true_am, y_pred_am, average='macro')
    f1 = f1_score(y_true_am, y_pred_am, average='macro')

    return cm, accuracy, precision, recall, f1
    
def log_metrics(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9], epoch=0, test=False, output_folder="log/"):
    cm, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred, labels)
    # Save metrics to file
    if test:
        with open(output_folder + f"log/test/metrics_test.txt", "a") as f:
            f.write(f"{epoch} {accuracy} {precision} {recall} {f1}\n")
    else:
        with open(output_folder + f"log/train/metrics_train.txt", "a") as f:
            f.write(f"{epoch} {accuracy} {precision} {recall} {f1}\n")
    # Print metrics
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    # save confusion matrix 
    plt.figure(figsize=(10,10))
    plt.title("Confusion matrix")
    sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if test:
        plt.savefig(output_folder + f"log/test/confusion_matrix_test_{epoch}.png")
    else:
        plt.savefig(output_folder + f"log/train/confusion_matrix_train_{epoch}.png")
    plt.clf()
    plt.close()

