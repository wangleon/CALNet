import torch
from model import *
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchmetrics
import datetime as datetime
from matplotlib import pyplot as plt
import random

starttime = datetime.datetime.now()
print("Start time:", starttime)

###torch 固定 随机数种子

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###利用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###load data
###positive data
T_lc = pd.read_csv("../Positive_and_Negative_Data/Positive_PreProcessed/LC/Positive_LC.csv", header=None)
T_GLS = pd.read_csv("../Positive_and_Negative_Data/Positive_PreProcessed/GLS/Positive_GLS.csv", header=None)
T_tic = pd.DataFrame(T_lc.loc[:, 0])
T_tic_GLS = T_GLS.loc[:, 0]
T_lcData = T_lc.loc[:, 1:]
T_GLSData = T_GLS.loc[:, 1:]

###negative data
F_lc = pd.read_csv("../Positive_and_Negative_Data/Negative_PreProcessed/LC/Negative_LC.csv", header=None)
F_GLS = pd.read_csv("../Positive_and_Negative_Data/Negative_PreProcessed/GLS/Negative_GLS.csv", header=None)
F_tic = pd.DataFrame(F_lc.loc[:, 0])
F_tic_GLS = F_GLS.loc[:, 0]
F_lcData = F_lc.loc[:, 1:]
F_GLSData = F_GLS.loc[:, 1:]

print(np.shape(F_GLSData))
print(np.shape(T_GLSData))
print(np.shape(F_lcData))
print(np.shape(T_lcData))

### Combine positive and negative data
lcData = pd.concat([T_lcData, F_lcData], axis=0)
GLSData = pd.concat([T_GLSData, F_GLSData], axis=0)
TICData = pd.concat([T_tic, F_tic], axis=0)

TIC_id = []
for i in range(len(TICData)):
    one = TICData[0].values
    split = one[i].split('-')
    TIC_id.append(int(split[2]))

lcData = torch.tensor(lcData.values, dtype=torch.float32)
GLSData = torch.tensor(GLSData.values, dtype=torch.float32)
TICData = torch.tensor(TIC_id, dtype=torch.int64)
print(lcData.shape)
print(GLSData.shape)
### create labels
labels = torch.cat((torch.ones(T_lcData.shape[0], dtype=torch.float32), 
                    torch.zeros(F_lcData.shape[0], dtype=torch.float32)), dim=0)
print(labels.shape)

###split train and test data
dataset = TensorDataset(lcData, GLSData, labels, TICData)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=128, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128, drop_last=True)

# for lc_batch, gls_batch, labels_batch, TIC_batch in test_loader:
#     for i in range(len(lc_batch)):
#         print(i)
#         fig, ax = plt.subplots(2, 1)
#         ax[0].plot(lc_batch[i].numpy(), label=labels_batch[i].item())
#         ax[1].plot(gls_batch[i].numpy(), label=labels_batch[i].item())
#         plt.legend()
#         plt.savefig("../data/test_data/{}_{}.png".format(TIC_batch[i].item(), labels_batch[i].item()))
#         plt.close()

## train model
model = CNN_CBAM_LSTM_model()
model = model.to(device)  # Move model to GPU if available
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
criterion = criterion.to(device)  # Move criterion to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)  # Adam optimizer

for epoch in range(30):  # Example: 10 epochs
    print("*************epoch:{}************".format(epoch+1))
    branch_start_time = datetime.datetime.now()
    print("Branch start time:", branch_start_time)
    ### begin training
    model.train()  # Set the model to training mode
    for lc_batch, gls_batch, labels_batch, TIC_batch in train_loader:
        lc_batch, gls_batch, labels_batch = lc_batch.to(device), gls_batch.to(device), labels_batch.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        # Forward pass
        outputs = model(lc_batch.unsqueeze(1), gls_batch.unsqueeze(1))  # Add channel dimension
        loss = criterion(outputs, labels_batch.long())  # Use long() to convert labels to the correct type for CrossEntropyLoss
        loss.backward()
        optimizer.step()
    
    branch_end_time = datetime.datetime.now()
    print("Branch end time:", branch_end_time)
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}') 
  
torch.save(model.state_dict(), 'CNN_CBAM_LSTM_model_dict.pth')  # Save the trained model
### begin testing
model.eval()  # Set the model to evaluation mode
test_acc = torchmetrics.Accuracy(task='binary', num_classes=2).to(device)
test_recall = torchmetrics.Recall(task='binary',num_classes=2).to(device)
test_precision = torchmetrics.Precision(task='binary',num_classes=2).to(device)
test_f1 = torchmetrics.F1Score(task='binary',num_classes=2).to(device)
size = len(test_loader.dataset)
num_batches = len(test_loader)
test_loss, test_correct = 0, 0
with torch.no_grad():
    for lc_batch, gls_batch, labels_batch, TIC_batch in test_loader:
        lc_batch, gls_batch, labels_batch = lc_batch.to(device), gls_batch.to(device), labels_batch.to(device)  # Move data to GPU if available
        outputs = model(lc_batch.unsqueeze(1), gls_batch.unsqueeze(1))  # Add channel dimension
        # outputs = outputs.permute(1,0)
        print(outputs)
        print(labels_batch)
        loss = criterion(outputs, labels_batch.long())
        test_loss += loss.item() * lc_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels_batch.long()).sum().item()
        
        # Update metrics
        test_acc.update(predicted, labels_batch.long())
        test_recall.update(predicted, labels_batch.long())
        test_precision.update(predicted, labels_batch.long())
        test_f1.update(predicted, labels_batch.long())
test_loss /= size
test_acc_value = test_acc.compute()
test_recall_value = test_recall.compute()
test_precision_value = test_precision.compute()
test_f1_value = test_f1.compute()
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc_value:.4f}, '
        f'Test Recall: {test_recall_value:.4f}, Test Precision: {test_precision_value:.4f}, '
        f'Test F1 Score: {test_f1_value:.4f}')

###清空计算对象
test_acc.reset()
test_recall.reset()
test_precision.reset()
test_f1.reset()

endtime = datetime.datetime.now()
print("End time:", endtime)
print("Total time:", endtime - starttime)
