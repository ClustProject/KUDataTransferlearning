# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:30:33 2023

@author: lee
"""


# =============================================================================
# Train notebook file
# =============================================================================
import torch
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
import config
import utils
import main_transfer as mt

SOURCE_DATASET = 'Computers'
MODE = 'TRAIN'

data_root_dir = f'./data/{SOURCE_DATASET}/'

train_x, train_y, test_x, test_y = utils.load_data(data_root_dir, model_name='LSTM_FCNs')  # shape=(num_of_instance, input_dims, time_steps)
        
split_ratio = 0.2
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split_ratio, shuffle=True)

input_size = train_x.shape[1]
num_classes = len(np.unique(train_y))


# normalization
new_dir_path = f'./scaler/{SOURCE_DATASET}/'
os.makedirs(new_dir_path, exist_ok=True)

scaler_x_path = f'./scaler/{SOURCE_DATASET}/minmax_scaler_x.pkl'
train_x, valid_x = utils.get_train_val_data(train_x, valid_x, scaler_x_path)

model_name = 'LSTM_FCNs'
model_params = config.model_config[model_name]

model_params['parameter']['input_size'] = input_size
model_params['parameter']['num_classes'] = num_classes
model_params['best_model_path'] = f'./ckpt/{SOURCE_DATASET}/lstm_fcn.pt'

data_source = mt.Transferlearning(model_params,'self')

data_source.build_model()

best_model = data_source.train_model(train_x, train_y, valid_x, valid_y)  # 모델 학습

os.makedirs(f'./ckpt/{SOURCE_DATASET}/', exist_ok=True)
data_source.save_model(best_model, best_model_path=model_params["best_model_path"])  # 모델 저장

print(num_classes)
# =============================================================================
# Test / Trasnfer
# =============================================================================

import config
import utils
import torch.nn as nn
TARGET_DATASET = 'ScreenType'
MODE = 'TEST'

# load raw data
data_root_dir = f'./data/{TARGET_DATASET}/'
train_x, train_y, test_x, test_y = utils.load_data(data_root_dir, model_name='LSTM_FCNs')  # shape=(num_of_instance, input_dims, time_steps)

input_size = train_x.shape[1]
num_classes = len(np.unique(train_y))

# normalization

split_ratio = 0.2
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split_ratio, shuffle=True)

scaler_x_path = f'./scaler/{TARGET_DATASET}/minmax_scaler_x.pkl'
train_x, valid_x = utils.get_train_val_data(train_x, valid_x, scaler_x_path)

model_name = 'LSTM_FCNs'
model_params = config.model_config[model_name]

model_params['parameter']['input_size'] = input_size
#model_params['parameter']['num_classes'] = 3
model_params['best_model_path'] = f'./ckpt/{SOURCE_DATASET}/lstm_fcn.pt' ##SOURCE DATSET 

data_target = mt.Transferlearning(model_params,'transfer')
data_target_self = mt.Transferlearning(model_params,'self')



## transfer 할때, num class 가 다르면 fine-tuning 및 fc layer 업데이트 하는 쪽으로.. 

# tt = data_target.build_model()
# in_features = tt.fc.in_features

# tt.fc = nn.Linear(in_features,2)

# for param in tt.parameters():
#     print(param)
#     param.requires_grad = False

# for name, param in tt.named_parameters():
#     print(name)
#     if name in ['fc.weight','fc.bias']:
#         param.requires_grad = True
#     else :
#         param.requires_grad = False

# for param in tt.parameters():
#     print(param.requires_grad)

#data_target.tuning_model(model_params['best_model_path'],freeze=True)
#data_target.tuning_model(model_params['best_model_path'],freeze=False)

# tt = data_target.build_model()
# tt.load_state_dict(torch.load(model_params['best_model_path']))

#best_model = data_target.train_model(train_x, train_y, valid_x, valid_y,'target')
best_model = data_target_self.train_model(train_x, train_y, valid_x, valid_y,'source')

#pred, acc = data_target.pred_data(test_x, test_y, best_model_path=model_params["best_model_path"])  # 예측

# # 예측 결과 확인
# print(f'** Performance of test dataset ==> Acc = {acc}')
# print(f'** Dimension of result for test dataset = {pred.shape}')
# print(pred.head())

