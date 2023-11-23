# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:29:09 2023

@author: lee
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from models.train_model import Train_Test
from models.lstm_fcn import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC

import warnings
warnings.filterwarnings('ignore')

class Transferlearning():
    def __init__(self, config, mode):
        """

        Parameters
        ----------
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mode = mode
        self.model_name = config['model']
        self.parameter = config['parameter']
        self.best_model_path = config['best_model_path']
        # build trainer
        self.trainer = Train_Test(config)
        
    def build_model(self):
        """
        
        Returns
        -------
        init_model : TYPE
            DESCRIPTION.

        """
        
        if self.mode == 'transfer' : # transfer model 학습 
            init_model = LSTM_FCNs(
                    input_size=self.parameter['input_size'],
                    num_classes=self.parameter['source_class'],
                    num_layers=self.parameter['num_layers'],
                    lstm_drop_p=self.parameter['lstm_drop_out'],
                    fc_drop_p=self.parameter['fc_drop_out']
                )
        
        else : ## target 자체를 학습시키는 모델 만듬 ##self
            init_model = LSTM_FCNs(
                    input_size=self.parameter['input_size'],
                    num_classes=self.parameter['num_classes'],
                    num_layers=self.parameter['num_layers'],
                    lstm_drop_p=self.parameter['lstm_drop_out'],
                    fc_drop_p=self.parameter['fc_drop_out']
                )
        
        return init_model
    
    def train_model(self,train_x, train_y, valid_x, valid_y,option='source'):
        """
        

        Parameters
        ----------
        train_x : TYPE
            DESCRIPTION.
        train_y : TYPE
            DESCRIPTION.
        valid_x : TYPE
            DESCRIPTION.
        valid_y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        train_loader = self.get_dataloader(train_x, train_y, self.parameter['batch_size'], shuffle=True)
        valid_loader = self.get_dataloader(valid_x, valid_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        if option == 'target' :
            init_model = self.tuning_model(self.best_model_path,freeze=self.parameter['freeze'])
        else :
            init_model = self.build_model()
        
        # train model
        dataloaders_dict = {'train': train_loader, 'val': valid_loader}
        best_model = self.trainer.train(init_model, dataloaders_dict)
        return best_model
        
        
        
    def save_model(self,best_model,best_model_path):
        """

        Parameters
        ----------
        best_model : TYPE
            DESCRIPTION.
        best_model_path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        torch.save(best_model.state_dict(), best_model_path)
        
    
    def pred_data(self,test_x, test_y, best_model_path):
        """
        """
        
        test_loader = self.get_dataloader(test_x, test_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))

        # get predicted classes
        pred_data = self.trainer.test(init_model, test_loader)

        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        if np.min(test_y) != 0:
            print('Set start class as zero')
            test_y = test_y - np.min(test_y)

        # calculate performance metrics
        acc = accuracy_score(test_y, pred_data)
        
        # merge true value and predicted value
        pred_df = pd.DataFrame()
        pred_df['actual_value'] = test_y
        pred_df['predicted_value'] = pred_data
        return pred_df, acc
    
    def get_dataloader(self, x_data, y_data, batch_size, shuffle):
        """
        
        """
        if np.min(y_data) != 0:
            print('Set start class as zero')
            y_data = y_data - np.min(y_data)

        # torch dataset 구축
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))

        # DataLoader 구축
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
        
    
    def tuning_model(self,best_model_path,freeze):
        # config 에 Source / Target dataset 정리       
        ## change / freeze / save
        
        # load best model
        init_model = self.build_model()
        init_model.load_state_dict(torch.load(best_model_path))
        
        if self.parameter['source_class'] != self.parameter['target_class'] :
            
            print('model fc layer output change')
            in_features = init_model.fc.in_features
            out_features = self.parameter['target_class']
            
            init_model.fc = nn.Linear(in_features,out_features)
        
        if freeze:
            for name, param in init_model.named_parameters():
                if name in ['fc.weight','fc.bias']:
                    param.requires_grad = True
                
                else :
                    param.requires_grad = False
                print(param.requires_grad)
        return init_model
    

# import pandas as pd
# import numpy as np
# from scipy.io.arff import loadarff
# import pickle

# DATASET = 'Computers'
# MODE = 'TRAIN'

# train = loadarff(f"./data/{DATASET}/{DATASET}_TRAIN.arff")
# test = loadarff(f"./data/{DATASET}/{DATASET}_TEST.arff")

# data_train = np.asarray([train[0][name] for name in train[1].names()])
# X_train = data_train[:-1].T.astype('float64')
# y_train = data_train[-1]

# data_test = np.asarray([test[0][name] for name in test[1].names()])
# X_test = data_test[:-1].T.astype('float64')
# y_test = data_test[-1]

# try:
#     y_train = y_train.astype('float64').astype('int64')
#     y_test = y_test.astype('float64').astype('int64')
# except ValueError:
#     y_train = y_train.astype(str)
#     y_test = y_test.astype(str)

# y_train 
# y_test

# X_train = X_train[:, np.newaxis,:]
# X_test = X_test[:, np.newaxis,:]

# with open(f'./data/{DATASET}/x_train.pkl', 'wb') as f:
#     pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)

# with open(f'./data/{DATASET}/x_test.pkl', 'wb') as f:
#     pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
    
# with open(f'./data/{DATASET}/y_train.pkl', 'wb') as f:
#     pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
        
# with open(f'./data/{DATASET}/y_test.pkl', 'wb') as f:
#     pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)

