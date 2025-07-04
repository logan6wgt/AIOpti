import os
os.environ['GRB_LICENSE_FILE'] = '/opt/conda/lib/python3.8/site-packages/gurobipy-11.0.0.dist-info/gurobi.lic'

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pandas import Index
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pywt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import torch.nn.functional as F
from PyEMD import EMD
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import Transformer
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
import time
from filterpy.kalman import KalmanFilter
from scipy.fftpack import fft, fftfreq
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


##
def load_data(sequence_length, input_size, output_size, batch_size):
    
    train_data = pd.read_csv('train_date.csv')
    test_data = pd.read_csv('val_data.csv')
    test_data_ori = pd.read_csv('test_data.csv')

    print("Shape of train_data after preprocessing:", test_data.shape)
   
    print(train_data.columns)

    test_data = test_data[train_data.columns]
    test_data_ori = test_data_ori[train_data.columns]
   
    scaler = MinMaxScaler()
   
    # Fit the scaler using only the train data
    scaler.fit(train_data)
    # Transform both train and test data
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    scaled_test_data_ori = scaler.transform(test_data_ori)

    # Extract feature range (min and max) for each feature
    train_data_df = pd.DataFrame(train_data)
    feature_ranges = {}
    for i, feature in enumerate(train_data_df.columns):
       data_min = float(scaler.data_min_[i])
       data_max = float(scaler.data_max_[i])
       feature_ranges[feature] = {"min": data_min, "max": data_max}

    # Write the feature ranges to a JSON file for later use
    with open('feature_ranges.json', 'w') as outfile:
       json.dump(feature_ranges, outfile)

    # transfer data to tensor
    scaled_train_data = torch.FloatTensor(scaled_train_data)
    scaled_test_data = torch.FloatTensor(scaled_test_data)
    train_data = torch.FloatTensor(train_data.values)
    scaled_test_data_ori = torch.FloatTensor(scaled_test_data_ori)

    def split_monthly_sequences(data_tensor, input_sequence_length, output_sequence_length, feature_index_for_label):
        sequences = []
        label = []
        total_intervals = data_tensor.shape[0]
    
        for i in range(0, total_intervals - input_sequence_length - output_sequence_length + 1, 6):
            input_sequence = data_tensor[i:i + input_sequence_length, :]
           
            output_sequence = data_tensor[i + input_sequence_length:i + input_sequence_length + output_sequence_length, -1]  
            label.append(output_sequence)

            sequences.append((input_sequence, output_sequence))
        
        return sequences, label

    daily_sequences, daily_label_train = split_monthly_sequences(scaled_train_data, sequence_length, output_size, input_size)

    daily_sequences_test, daily_label_test = split_monthly_sequences(scaled_test_data, sequence_length, output_size, input_size)

    daily_sequences_train, label_train = split_monthly_sequences(train_data, sequence_length, output_size, input_size)
    daily_sequences_test_real, label_test = split_monthly_sequences(scaled_test_data_ori, sequence_length, output_size, input_size)

    return daily_sequences, daily_sequences_test, daily_sequences_test_real, label_train, label_test, scaled_train_data, scaled_test_data, test_data, train_data

# split_daily_sequences
batch_size = 512 
input_size = 10  
output_size = 96  
sequence_length = 144 

daily_sequences, daily_sequences_test, daily_sequences_test_real, label_train, label_test, scaled_train_data, scaled_test_data, test_data, train_data\
    = load_data(sequence_length, input_size, output_size, batch_size)

print('train dataset:', len(daily_sequences))
print('test dataset:', len(daily_sequences_test))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class SELU(nn.Module):
    def forward(self, x):
        return F.selu(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
class ReLU(nn.Module):
    def forward(self, x):
        return F.relu(x)

class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, self.alpha)

#### model

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        scores = self.attention_weights(x)
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector

class ResidualBlockRES(nn.Module):
    def __init__(self, input_size, hidden_size, activation='elu', dropout_rate=0.2):
        super(ResidualBlockRES, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        if input_size != hidden_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size)
            )
        else:
            self.shortcut = nn.Identity()
    
    def _get_activation(self, activation):
        if activation == 'swish':
            return nn.SiLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'mish':
            return nn.Mish()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.activation(out)
        
        return out

class ResNetLSTM_MultiBranch(nn.Module):
    def __init__(self, input_size=11, seq_length=144, hidden_size=256, lstm_hidden_size=256, output_size=96, num_blocks=4, dropout_rate=0.1, activation = 'elu'):
        super(ResNetLSTM_MultiBranch, self).__init__()
        self.seq_length = seq_length
        
       
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(ResidualBlockRES(input_size * seq_length, hidden_size, activation, dropout_rate))
            else:
                layers.append(ResidualBlockRES(hidden_size, hidden_size, activation, dropout_rate))
        self.resnet = nn.Sequential(*layers)
        
        
        self.fc_resnet = nn.Linear(hidden_size, lstm_hidden_size * seq_length)
        
        
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, num_layers=1, batch_first=True, dropout=dropout_rate, bidirectional=False)
        
       
        self.value_head = nn.Linear(lstm_hidden_size, output_size)
        self.variance_head = nn.Linear(lstm_hidden_size, output_size)
        
        self.fusion = nn.Linear(output_size * 2, output_size)
        
        self.attention = Attention(output_size * 2)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        batch_size, seq_length, input_size = x.size()
        assert seq_length == self.seq_length, "error"
        
        # [batch_size, seq_length * input_size]
        x_flat = x.view(batch_size, -1)
        
        resnet_out = self.resnet(x_flat)
        
        resnet_mapped = self.fc_resnet(resnet_out)
        resnet_seq = resnet_mapped.view(batch_size, seq_length, -1)
        
        lstm_out, _ = self.lstm(resnet_seq)
        lstm_last = lstm_out[:, -1, :]
        
        value_output = self.value_head(lstm_last)
        variance_output = self.variance_head(lstm_last)

        combined_output = torch.cat((value_output, variance_output), dim=-1)
        
        context_vector = self.attention(combined_output.unsqueeze(1))
        final_output = self.fusion(context_vector)
       
        return final_output

if __name__ == "__main__":

    with open('feature_ranges.json', 'r') as infile:
        feature_ranges = json.load(infile)
    
    last_feature = list(feature_ranges.keys())[-1]

    data_min = feature_ranges[last_feature]['min']
    data_max = feature_ranges[last_feature]['max']

    # build DataLoader
    from torch.utils.data import DataLoader
    batch_size = batch_size

    class SequenceDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
        
            return self.sequences[idx]

    dataset = SequenceDataset(daily_sequences)
    dataset_test = SequenceDataset(daily_sequences_test)
    dataset_test_real = SequenceDataset(daily_sequences_test_real)

    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    loader_test_real = DataLoader(dataset_test_real, batch_size=batch_size, shuffle=False)

    model =  ResNetLSTM_MultiBranch()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    mse_loss_func = torch.nn.MSELoss()
    mae_loss_func = torch.nn.L1Loss()
    def rmse_loss_func(y_pred, y_true):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    def mape_loss_func(y_pred, y_true, epsilon=1e-8):
        y_true_abs = torch.abs(y_true)
        y_pred_abs = torch.abs(y_pred)
        return torch.mean(torch.abs((y_true_abs - y_pred_abs) / torch.clamp(y_true_abs, min=epsilon))) * 100
    def smape_loss_func(y_pred, y_true):
        return 100 * torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true)))
    def mase_loss_func(y_pred, y_true, seasonal_period=1):
        n = y_true.size(0)
        d = torch.mean(torch.abs(y_true[seasonal_period:] - y_true[:-seasonal_period]))
        errors = torch.abs(y_true - y_pred)
        return torch.mean(errors / d)
    def medae_loss_func(y_pred, y_true):
        return torch.median(torch.abs(y_pred - y_true))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    epochs = 1000
 
    total_loss = 0.0
    total_loss1 = 0.0
    loss_history = []
    loss_history1 = []
    regret_history = []
    regret_unam_history = []
    mse_history = []
    mape_history = []
    mae_history = []
    smape_history = []
    mase_history = []
    medae_history = []
    predictions_history = []
    predictions_history_ori = []
    real_value = []

    def inverse_transform(data, data_min, data_max):
        return data * (data_max - data_min) + data_min
        
    # save path
    save_path0 = 'price_ResLSTM.pth'

    patience = 50
    best_regret = float('inf')
    counter = 0  
    
    for epoch in range(epochs):
        total_loss = 0.0  
        for i, (batch_tensors, batch_labels) in enumerate(loader_train):

            if torch.cuda.is_available():
                batch_tensors, batch_labels = batch_tensors.cuda(), batch_labels.cuda()
           
            seqs = batch_tensors
            labels = batch_labels
            y_pred = model(seqs)

            loss = mse_loss_func(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() 

        average_loss = total_loss / len(loader_train)
        loss_history.append(average_loss)  
        print(f'Epoch {epoch+1}, Loss: {average_loss}')

        # update lr
        scheduler.step()

        model.eval()  
        with torch.no_grad():  
            mse_loss = 0
            mae_loss = 0
            mape_loss = 0  
            rmse = 0
            smape = 0
            medae = 0
            all_preds = []
            all_labels = []
            for i, (batch_tensors_test, batch_labels_test) in enumerate(loader_test):
            # cuda
                if torch.cuda.is_available():
                    batch_tensors_test, batch_labels_test = batch_tensors_test.cuda(), batch_labels_test.cuda()

                predictions = model(batch_tensors_test)

                all_preds.append(predictions.cpu().numpy())
                all_labels.append(batch_labels_test.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            all_preds = inverse_transform(all_preds, data_min, data_max)
            all_labels = inverse_transform(all_labels, data_min, data_max)

            all_preds = torch.tensor(all_preds, dtype=torch.float32)
            all_labels = torch.tensor(all_labels, dtype=torch.float32)

            mse_loss = mse_loss_func(all_preds, all_labels).item()
            mape_loss = mape_loss_func(all_preds, all_labels).item()
            mae_loss = mean_absolute_error(all_labels.numpy(), all_preds.numpy())
            rmse = rmse_loss_func(all_preds, all_labels).item()
            smape = smape_loss_func(all_preds, all_labels).item()
            medae = medae_loss_func(all_preds, all_labels).item()

            if mse_loss < best_regret:
                best_regret = mse_loss
                counter = 0  
                
                torch.save(model.state_dict(), save_path0)
            else:
                counter += 1  
            if counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break  
            
        print(f'Epoch {epoch + 1}/{epochs} - Test MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, MAPE: {mape_loss:.4f}%,  RMSE: {rmse:.4f}, SMAPE: {smape:.2f}%, MedAE: {medae:.4f}')
       
        model.train()  
        
        torch.cuda.empty_cache()
    
    
    model.load_state_dict(torch.load('price_ResLSTM.pth'))
    model.eval()
   
    with torch.no_grad():  
        mse_loss = 0
        mae_loss = 0
        mape_loss = 0  
        rmse = 0
        smape = 0
        medae = 0
        all_preds = []
        all_labels = []
        for i, (batch_tensors_test, batch_labels_test) in enumerate(loader_test_real):
        # cuda
            if torch.cuda.is_available():
                batch_tensors_test, batch_labels_test = batch_tensors_test.cuda(), batch_labels_test.cuda()

            predictions = model(batch_tensors_test)

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(batch_labels_test.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        all_preds = inverse_transform(all_preds, data_min, data_max)
        all_labels = inverse_transform(all_labels, data_min, data_max)

        all_preds = torch.tensor(all_preds, dtype=torch.float32)
        all_labels = torch.tensor(all_labels, dtype=torch.float32)

        mse_loss = mse_loss_func(all_preds, all_labels).item()
        mape_loss = mape_loss_func(all_preds, all_labels).item()
        mae_loss = mean_absolute_error(all_labels.numpy(), all_preds.numpy())
        rmse = rmse_loss_func(all_preds, all_labels).item()
        smape = smape_loss_func(all_preds, all_labels).item()
        medae = medae_loss_func(all_preds, all_labels).item()

    print(f'Test real - Test MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, MAPE: {mape_loss:.4f}%,  RMSE: {rmse:.4f}, SMAPE: {smape:.2f}%, MedAE: {medae:.4f}')
    print(data_max)
    print(data_min)

