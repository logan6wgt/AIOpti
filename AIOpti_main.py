import os
os.environ['GRB_LICENSE_FILE'] = '/opt/conda/lib/python3.8/site-packages/gurobipy-11.0.0.dist-info/gurobi.lic'

import os
import csv
import time
import math
import json
import pickle

import numpy as np
import pandas as pd
from pandas import Index

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import gurobipy as gp
from gurobipy import GRB, quicksum

import pyepo
from pyepo.model.grb import optGrbModel

import matplotlib.pyplot as plt
import seaborn as sns

from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from PyEMD import EMD

from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from scipy.fftpack import fftfreq

from filterpy.kalman import KalmanFilter

import statsmodels.api as sm

# import ML model
from AIOpti_MLmodel import ResidualBlockRES
from AIOpti_MLmodel import ResNetLSTM_MultiBranch

import time

start_time = time.time()

## data read for optimization mpdel

deltaT = 0.25

# parameters of energy storage using EVs

smax = 57.5 * 20 #
bcamax = 36
bdcmax = 36
inta_EV = 0.98
DOD_EV = 0.8
estmax = 7 * 20# Max Charge and discharge power

# parameters of battery energy storage

bsmax = 13.5 * 20#"kWh"
bbcamax = 96
bbdcmax = 96
DOD = 0.9
inta_BE = 0.98
pchamax = 5 * 20#"kW"
pdchamax = 5 * 20#"kW"

##

###
# second stage
# optimization model

# read the initial solutions from model stage 1
all_sheets = pd.read_excel('result_M1.xlsx', sheet_name=None)

myPG  = all_sheets['myPG'].to_numpy().ravel()    # 1D
myPW  = all_sheets['myPW'].to_numpy().ravel()
myPS  = all_sheets['myPS'].to_numpy().ravel()
myPGP = all_sheets['myPGP'].to_numpy().ravel()

myPPV = all_sheets['myPPV'].to_numpy()           # 2D: shape (D, T)
myLTO = all_sheets['myLTO'].to_numpy()

##
def load_data(sequence_length, input_size, output_size, batch_size):

    train_data = pd.read_csv('train_date.csv')
    test_data = pd.read_csv('val_data.csv')
    test_data_ori = pd.read_csv('test_data.csv')

    print("Shape of train_data after preprocessing:", test_data.shape)
    # 检查列名
    print(train_data.columns)

    test_data = test_data[train_data.columns]
    test_data_ori = test_data_ori[train_data.columns]
   
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    
    # Fit the scaler using only the train data
    scaler.fit(train_data)
    # Transform both train and test data
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    scaled_test_data_ori = scaler.transform(test_data_ori)

    # Extract feature range (min and max) for each feature
    train_data_df = pd.DataFrame(train_data)
    feature_ranges_end = {}
    for i, feature in enumerate(train_data_df.columns):
       data_min = float(scaler.data_min_[i])
       data_max = float(scaler.data_max_[i])
       feature_ranges_end[feature] = {"min": data_min, "max": data_max}

    # Write the feature ranges to a JSON file for later use
    with open('feature_ranges_end.json', 'w') as outfile:
       json.dump(feature_ranges_end, outfile)

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

# parameter setting
batch_size = 512  # batch_size
input_size = 10  # position_label
output_size = 96  # output_size
sequence_length = 144 # seq_length

daily_sequences, daily_sequences_test, daily_sequences_test_real, label_train, label_test, scaled_train_data, scaled_test_data, test_data, train_data\
    = load_data(sequence_length, input_size, output_size, batch_size)

print('train dataset:', len(daily_sequences))
print('test dataset:', len(daily_sequences_test))

# optimization model
class myOptModel(optGrbModel):
    def __init__(self, weights, myPG, myPW, myPS, myPGP, myPPV, myLTO):
        self.weights = np.array(weights)
        self.num_item = 96  # T = 96
        self.D = 1
        self.myPG = myPG
        self.myPW = myPW
        self.myPS = myPS
        self.myPGP = myPGP
        self.myPPV = myPPV
        self.myLTO = myLTO
        super().__init__()

    def _getModel(self):

        # ceate a model
        m = gp.Model()

        # set
        T = 96
        D = 1
       
        M = 100000000

        # varibles
        x = {}
        for t in range(T):
            x[t] = m.addVar(lb=-GRB.INFINITY, name=f"x_{t}")


        # model sense
        m.modelSense = GRB.MAXIMIZE

        m.setParam('MIPGap', 0.01)  # set gap(1%) for acceleration

        # constraints

        # power variables
        PG = {}  # grid
        PP = {}  # individual battery,charge
        PPD = {}  # individual battery,discharge
        BES = {}  # energy stored in individual battery
        PST = {}  # energy storage(EVs), charge
        PSTD = {}  # energy storage(EVs), discharge
        EVES = {}

        # gas variable
        for t in range(T):
            PG[t] = m.addVar(lb = 0)

        for d in range(D):
            for t in range(T):
                PP[d, t] = m.addVar(lb=0)
                PPD[d, t] = m.addVar(lb=0)
                BES[d, t] = m.addVar(lb=0)
                PST[d, t] = m.addVar(lb=0)
                PSTD[d, t] = m.addVar(lb=0)
                EVES[d, t] = m.addVar(lb=0)

        # binary variables
        BBCA = {}  # for battery charge
        BBDC = {}  # for battery discharge
        BCA = {}  # for EV charge
        BDC = {}  # for EV discharge
        BP = {}  
        for d in range(D):
            for t in range(T):
                BBCA[d, t] = m.addVar(vtype=GRB.BINARY)
                BBDC[d, t] = m.addVar(vtype=GRB.BINARY)
                BCA[d, t] = m.addVar(vtype=GRB.BINARY)
                BDC[d, t] = m.addVar(vtype=GRB.BINARY)
        for t in range(T):
            BP[t] = m.addVar(vtype=GRB.BINARY)

        # power sell
        PSE = {}
        for t in range(T):
            PSE[t] = m.addVar()

        # # constraints
        for t in range(T):
            m.addConstr(PG[t] + self.myPW[t] + self.myPS[t] + self.myPGP[t] + self.myPPV[0, t] + quicksum(PSTD[d, t] + PPD[d, t] for d in range(D)) == PSE[t] + quicksum(PST[d, t] + PP[d, t]  for d in range(D)) 
            + 0.001 * self.myLTO[0, t] )

        # battery system
        # battery system
        for d in range(D):
            for t in range(T):
                m.addConstr(PPD[d, t] <= BBDC[d, t] * pdchamax)
                m.addConstr(PP[d, t] <= BBCA[d, t] * pchamax)
                m.addConstr(PPD[d, t] >= 0)
                m.addConstr(PP[d, t] >= 0)
                m.addConstr(BBCA[d, t] + BBDC[d, t] <= 1)

        for d in range(D):
            m.addConstr(BES[d, 0] == bsmax)
            m.addConstr(PP[d, 0] == 0)
            m.addConstr(PPD[d, 0] == 0)

        for d in range(D):
            for t in range(1, T):
                m.addConstr(BES[d, t] == BES[d, t - 1] + deltaT * (PP[d, t] * inta_BE - PPD[d, t] / inta_BE))
                m.addConstr(BES[d, t] <= bsmax)
                m.addConstr(BES[d, t] >= (1 - DOD) * bsmax)
        for d in range(D):
            m.addConstr(quicksum(BBCA[d, t] for t in range(T)) <= bbcamax)
            m.addConstr(quicksum(BBDC[d, t] for t in range(T)) <= bbdcmax)
            m.addConstr(BES[d, T - 1] == bsmax)

        # energy storage using EVs
        for d in range(D):
            for t in range(1, T):
                if t < 28:
                    m.addConstr(EVES[d, t] == EVES[d, t - 1] + deltaT * (PST[d, t] * inta_EV - PSTD[d, t] / inta_EV))
                    m.addConstr(EVES[d, t] >= (1 - DOD_EV) * smax)
                    m.addConstr(EVES[d, t] <= smax)

                else:
                    m.addConstr(EVES[d, t] == EVES[d, t - 1])
                    m.addConstr(BDC[d, t] == 0)
                    m.addConstr(BCA[d, t] == 0)

        for d in range(D):
            m.addConstr(quicksum(BCA[d, t] for t in range(T)) <= bcamax)
            m.addConstr(quicksum(BDC[d, t] for t in range(T)) <= bdcmax)
            m.addConstr(EVES[d, 0] == smax)
            m.addConstr(EVES[d, 27] >= smax * 0.7 )
            m.addConstr(PST[d, 0] == 0 )
            m.addConstr(PSTD[d, 0] == 0 )
       
        for d in range(D):
            for t in range(1, 27):
                m.addConstr(quicksum(BCA[d, tt] for tt in range(t, t + 2)) >= 2 * (BCA[d, t] - BCA[d, t - 1]))
                m.addConstr(quicksum(BDC[d, tt] for tt in range(t, t + 2)) >= 2 * (BDC[d, t] - BDC[d, t - 1]))
               
        for d in range(D):
            for t in range(T):
                m.addConstr(PSTD[d, t] >= 0)
                m.addConstr(PST[d, t] >= 0)
                m.addConstr(PSTD[d, t] <= BDC[d, t] * estmax )
                m.addConstr(PST[d, t] <= BCA[d, t] * estmax )
                m.addConstr(BCA[d, t] + BDC[d, t] <= 1)

        for t in range(T):
            m.addConstr(PSE[t] <= BP[t] * 100000)
            m.addConstr(PSE[t] >= BP[t] * 0.5)
            m.addConstr(PG[t] >= (1 - BP[t]) * 0.5)
            m.addConstr(PG[t] <= (1 - BP[t]) * 100000)
            m.addConstr(x[t] == (PSE[t] - PG[t])*0.25*0.001)
           
        return m, x


if __name__ == "__main__":

    num_data_train = len(daily_sequences)
    num_data_test = len(daily_sequences_test)
    num_fea = 11
    num_output = output_size

    weights = np.ones((num_data_train, num_output))
    weights_test = np.ones((num_data_test, num_output))

    print('weights shape:', weights.shape)
    print('weights_test shape:', weights_test.shape)

    # to obtain input sequence(feature data)
    def data_fea(sequences):
        x = []      
        for seq in sequences:           
            input_seq = seq[0]
            x.append(input_seq)
        x_numpy_list = [tensor.numpy() for tensor in x]
        x = np.array(x_numpy_list)
       
        return x

    x_train = data_fea(daily_sequences)
    x_test = data_fea(daily_sequences_test)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # to obtain output sequence(label data)
    def data_cost(sequences):
        x = []       
        for seq in sequences:           
            output_seq = seq[1]

            x.append(output_seq)
        x_numpy_list = [tensor.numpy() for tensor in x]
        x = np.array(x_numpy_list)
       
        return x

    c_train = data_cost(daily_sequences)
    c_test = data_cost(daily_sequences_test)
   
    with open('feature_ranges_end.json', 'r') as infile:
        feature_ranges_end = json.load(infile)
    
    last_feature = list(feature_ranges_end.keys())[-1]

    data_min = feature_ranges_end[last_feature]['min']
    data_max = feature_ranges_end[last_feature]['max']

    def inverse_transform(data, data_min, data_max):
        return data * (data_max - data_min) + data_min
   
    c_train = inverse_transform(c_train, data_min, data_max)
    c_test = inverse_transform(c_test, data_min, data_max)

    print('c_train shape:', c_train.shape)
    print('c_test shape:', c_test.shape)

    optmodel = myOptModel(weights, myPG, myPW, myPS, myPGP, myPPV, myLTO)

    # build optDataset
    from pyepo.data.dataset import optDataset

    dataset_train = optDataset(optmodel, x_train, c_train)
    dataset_test = optDataset(optmodel, x_test, c_test)

    # build DataLoader
    from torch.utils.data import DataLoader
    batch_size = batch_size
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # prediction model
    model =  ResNetLSTM_MultiBranch()
  
    # load saved initial parameters for ML model
    model.load_state_dict(torch.load('price_ResLSTM.pth'))
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # spo+
    spop = pyepo.func.SPOPlus(optmodel, processes=0)

    # Differentiable Perturbed Optimizer
    # init ptb solver
    ptb = pyepo.func.perturbedOpt(optmodel, n_samples=20, sigma=1.0, processes=0)
    # set loss
    criterion_ptb = nn.L1Loss()

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

    # loss function
    def ptbl1(cp, w):
        # perturbed optimizer
        we = ptb(cp)
        # loss
        loss = l1(we, w)
        return loss

    # Perturbed Fenchel-Young Loss
    # init pfyl loss
    pfy = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=20, sigma=0.5, processes=0)

    # Differentiable Black-Box Optimizer
    # init dbb solver
    dbb = pyepo.func.blackboxOpt(optmodel, lambd=20, processes=0)
    # set loss
    criterion_dbb = nn.L1Loss()

    # Noise Contrastive Estimation
    nce = pyepo.func.NCE(optmodel, processes=0, solve_ratio=0.5, dataset=dataset_train)

    # # Maximum A Posterior
    cmap = pyepo.func.contrastiveMAP(optmodel, processes=0, solve_ratio=0.5, dataset=dataset_train)

    # Listwise Learning To Rank
    lsltr = pyepo.func.listwiseLTR(optmodel, processes=0, solve_ratio=0.5, dataset=dataset_train)
   
    # # Pairwise Learning To Rank
    prltr = pyepo.func.pairwiseLTR(optmodel, processes=0, solve_ratio=0.5, dataset=dataset_train)

    # # Pointwise Learning To Rank
    ptltr = pyepo.func.pointwiseLTR(optmodel, processes=0, solve_ratio=0.5, dataset=dataset_train)

    epochs = 100
    total_loss = 0.0
    total_loss1 = 0.0
    loss_history = []
    loss_history1 = []
    regret_history = []
    regret_unam_history = []
    mse_history = []
    predictions_history = []
    real_value = []
    
    mse_history = []
    mape_history = []
    mae_history = []
    smape_history = []
    mase_history = []
    medae_history = []

    # save path
    save_path0 = 'spo_pretrain_ResLSTM.pth'
    
    # training parameters
    patience = 10  
    best_regret = float('inf')
    counter = 0  # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # path for csv 
    csv_file_path = 'regret_values_spo_pretrain_ResLSTM.csv'

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Regret"])  

    for epoch in range(epochs):
        total_loss = 0
        total_loss1 = 0
        
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data  # feat, cost, sol, obj
            # print('c:', c.shape)
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward
            cp = model(x)
            # print('cp:', cp.shape)

            cp = cp* (data_max - data_min) + data_min

            # spo+ loss
            single_loss = spop(cp, c, w, z)

            # # black-box optimizer
            # wp = dbb(cp)
            # # objective value
            # zp = (wp * c).sum(1).view(-1, 1)
            # # regret loss
            # single_loss = criterion_dbb(zp, z)

            # pfy
            # single_loss = pfy(cp, w)

            # noise contrastive estimation loss
            # single_loss = nce(cp, w)
            # single_loss = cmap(cp, w)
            
            # LTR
            # single_loss = lsltr(cp, c)
            # single_loss = prltr(cp, c)
            # single_loss = ptltr(cp, c)

            # backward
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()
            
            total_loss += single_loss.item()  

        average_loss = total_loss / len(loader_train)

        loss_history.append(average_loss)

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.8f}')
        
        model.eval()  

        with torch.no_grad():  

            regret = 0
            regret_all = []
            predictions_history = []
            real_value = []
    
            for data in loader_test:
                x_test, c_test, w_test, z_test = data
                if torch.cuda.is_available():
                    x_test, c_test, w_test, z_test = x_test.cuda(), c_test.cuda(), w_test.cuda(), z_test.cuda()
                predictions = model(x_test)

                predictions = inverse_transform(predictions, data_min, data_max)
                
                predictions_history.append(predictions)
                real_value.append(c_test)

                # optDataset
                eva_test = optDataset(optmodel, x_test.cpu(), predictions.cpu())
                w_spo_list = [data[2] for data in eva_test]
                w_spo_combined = torch.cat(w_spo_list, dim=0).view(-1, predictions.shape[1]).to(device)

                regret_batch = z_test.view(-1).cpu().numpy() - (c_test * w_spo_combined).sum(dim=1).cpu().numpy()
                regret_all.append(regret_batch)

            regret_all = np.concatenate(regret_all)
            current_regret = regret_all.mean()

        print('Regret:', current_regret)
        print(data_max)
        print(data_min)

        # save regret
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, current_regret])

        model.train()  
       
        if current_regret < best_regret:
            best_regret = current_regret
            counter = 0  
            
            torch.save(model.state_dict(), save_path0)
        else:
            counter += 1  
    
        if counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break  
        
        torch.cuda.empty_cache()

end_time = time.time()
print("training time：%.4f s" % (end_time - start_time))
