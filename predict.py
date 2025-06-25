import torch
from model import *
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime as datetime

starttime = datetime.datetime.now()
print("Start time:", starttime)
###利用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###load model
# model = torch.load('./CNN_CBAM_LSTM_model.pth', weights_only=False)

model = CNN_CBAM_LSTM_model()
model.load_state_dict(torch.load('./CNN_CBAM_LSTM_model_dict.pth'))
model.to(device)

###load data
###positive data
# T_lc = pd.read_csv("../Predict_Data/S88_processedData/LC/s088.csv", header=None)
# T_GLS = pd.read_csv("../Predict_Data/S88_processedData/GLS/s088.csv", header=None)
T_lc = pd.read_csv("../Predict_Data/processedData/LC/s091.csv", header=None)
T_GLS = pd.read_csv("../Predict_Data/processedData/GLS/s091.csv", header=None)
T_tic = T_lc.loc[:, 0]
T_tic_GLS = T_GLS.loc[:, 0]
T_lcData = T_lc.loc[:, 1:]
T_GLSData = T_GLS.loc[:, 1:]
print(np.shape(T_lcData))
print(np.shape(T_GLSData))
print(np.shape(T_tic))

single_tic = list(set(T_tic))
print(len(single_tic))

TIC_id = []
for i in range(len(T_tic)):
    split = T_tic[i].split('-')
    TIC_id.append(int(split[2]))
    
lcData = torch.tensor(T_lcData.values, dtype=torch.float32)
GLSData = torch.tensor(T_GLSData.values, dtype=torch.float32)
TICData = torch.tensor(TIC_id, dtype=torch.int64)

dataset = TensorDataset(lcData, GLSData, TICData)
pred_loader = DataLoader(dataset, batch_size=128)
print("length of pred_loader:", len(pred_loader))

###predict
model.eval()
predictions = []
with torch.no_grad():
    i = 0
    for lc_batch, gls_batch, tic_batch in pred_loader:
        print(f"Processing batch {i + 1}")
        lc_batch = lc_batch.unsqueeze(1).to(device)  # Add channel dimension
        gls_batch = gls_batch.unsqueeze(1).to(device)  # Add channel dimension
        outputs = model(lc_batch, gls_batch)
        _, predicted = torch.max(outputs, 1)
        tic_batch = tic_batch.numpy()
        # predictions.extend(zip(tic_batch, outputs.cpu().numpy()))
        predictions.extend(zip(tic_batch, predicted.cpu().numpy()))
        i += 1

# print("Predictions:", predictions)
print(type(predictions))
print("Number of predictions:", len(predictions))


###plot EB light curve
N_EB = 0
EB_tic = []
for i in range(len(predictions)):
    prob = predictions[i][1]
    # if prob[1] >= 0.9:
    if prob == 1: 
        filename = T_tic[i]
        EB_tic.append(T_tic[i])
print("Number of predicted EBs:", len(EB_tic))
EB_Single_tic = list(set(EB_tic))
print("Number of unique predicted EBs:", len(EB_Single_tic))

for i in range(len(EB_Single_tic)):
    filename = EB_Single_tic[i]
    tic = int(EB_Single_tic[i].split('-')[2])
    path = '/data/NAS_TESS_LC/tess/lc/s091/' + filename
    print(path)
    with fits.open(path) as hdu:
        # print(hdu.info())
        data = hdu[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']
        q_lst = data['QUALITY']
        m = q_lst == 0
        time = time[m]
        flux = flux[m]
        m2 = ~np.isnan(flux)
        time = time[m2]
        flux = flux[m2]
        
        ###plot
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(time, flux, 'o', color='C0', ms=2, alpha=1, label=f'TIC {tic}')
        ax.set_title(f'Predicted EB Light Curve for TIC {tic}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.legend()
        fig.savefig(f'../Predict_Data/Pred_EB_LC/s091/TIC_{tic}.png', bbox_inches='tight', dpi=100)
        plt.close()