import pandas as pd
import numpy as np
import torch
import torch.nn as nn


#### Define the CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


### model structure
class CNN_CBAM_LSTM_model(nn.Module):
    def __init__(self):
        super(CNN_CBAM_LSTM_model, self).__init__()
        self.maxpool = nn.MaxPool1d(4)
        self.lstm1 = nn.LSTM(256, 128, 2, batch_first=True) ###(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 2, batch_first=True)
        
        self.lcbranch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            CBAM(64, reduction=8),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            CBAM(128, reduction=8),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        
        self.glsbranch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            CBAM(32, reduction=8),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            CBAM(64, reduction=8),
            nn.MaxPool1d(4),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(7936, 1024),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )
        
    def forward(self, x1, x2):
        x1 = self.lcbranch(x1)
        x1 = x1.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, features) for LSTM
        x1, _ = self.lstm1(x1)
        x1 = nn.Flatten()(x1)
        # print("LC Shape after Flatten:", x1.shape)
        
        x2 = self.glsbranch(x2)
        x2 = x2.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, features) for LSTM
        x2, _ = self.lstm2(x2)
        x2 = nn.Flatten()(x2)
        x2 = x2.repeat(1, 2)
        x2 = nn.Flatten()(x2)
        # print("GLS Shape after repeat:", x2.shape)
        
        ###combine
        x = x1 + x2
        x = self.fc2(x)
        # print("Output Shape:", x.shape)
        
        return nn.functional.softmax(x, dim=1)  # Return both softmax output and raw logits

if __name__ == "__main__":
    # Example usage
    model = CNN_CBAM_LSTM_model()
    # lcData = torch.randn(128, 1, 4000)  # Example LC data
    # GLSData = torch.randn(128, 1, 1000)   # Example GLS data
    # output = model(lcData, GLSData)
    # print("Output:", output)
    # print("Output shape:", output.shape)






