import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt5
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.onnx
device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 

import math

class DiabetesDataset(Dataset):
    def __init__(self,path):
        X = np.empty((6,))
        Y = np.empty((1,))
        with open(path) as F:
            read = csv.DictReader(F)
            for Line in read:
                if(Line['level'] == 'I'):Line['level'] = 1
                if(Line['level'] == 'II'):Line['level'] = 2
                if(Line['level'] == 'III'):Line['level'] = 3
                if(Line['level'] == 'IV'):Line['level'] = 4
                if(Line['level'] == 'V'):Line['level'] = 5
                if(Line['level'] == 'VI'):Line['level'] = 6
                if(Line['level'] == 'VII'):Line['level'] = 7
                Y = np.row_stack((Y,np.array(float(Line['level']))))
                del Line['num']
                A = []
                for x in Line.keys():A.append(float(Line[x]))
                X = np.row_stack((X,np.array(A[:-1],dtype = float)))
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y).float()
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len


custom_dataset = MyDataSet('data.csv')

train_size = int(len(custom_dataset) * 0.7)
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

