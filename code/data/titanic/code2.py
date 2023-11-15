import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt5
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.onnx
device = torch.device("cuda:0")  # you can continue going on here, like cuda:1- cuda:2....etc. 

import math

class DiabetesDataset(Dataset):
    def __init__(self,path):
        X = np.empty((3,))
        Y = np.empty((1,))
        with open(path) as F:
            read = csv.DictReader(F)
            for Line in read:
                if('Survived' in Line.keys()):
                    Y = np.row_stack((Y,np.array([(math.e / (math.e + 1)) if (float(Line['Survived']) == 1) else (1 / (math.e + 1))])))
                else:
                    Y = np.row_stack((Y,np.array(float(Line['PassengerId']))))
                del Line['PassengerId']   
                del Line['Name']
                del Line['Ticket']
                del Line['Fare']
                del Line['Embarked']
                if('Survived' in Line.keys()):
                    del Line['Survived']
                Line['Cabin'] = float(ord(Line['Cabin'][0]) - ord('A') + 1) if (len(Line['Cabin']) > 0) else 0
                Line['Sex'] = 1 if Line['Sex'] == 'male' else 0
                Line['Age'] = 0 if Line['Age'] == '' else Line['Age']
                # Line['Embarked'] = 1 if Line['Embarked'] == 'C' else 2 if Line['Embarked'] == 'S' else 3
                A = []
                A.append(float(Line['Sex']))
                A.append(float(Line['Age']))
                A.append(float(Line['Cabin']))
                X = np.row_stack((X,np.array(A,dtype = float)))
        X = np.delete(X,0,0)
        Y = np.delete(Y,0,0)
        self.len = X.shape[0]
        # print(X)
        # print(Y)
        np.savetxt("code/data/titanic/train_del.csv",X,delimiter = ',')
        np.savetxt("code/data/titanic/train_ans.csv",Y,delimiter = ',')
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y).float()
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

testset = DiabetesDataset('code/data/titanic/test.csv')
dataset = DiabetesDataset('code/data/titanic/train.csv')
train_loader = DataLoader(dataset = dataset,batch_size = 150,shuffle = True,num_workers = 3)

# print(x_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(3,2)
        self.linear2 = torch.nn.Linear(2,1)
        self.active = torch.nn.Sigmoid() 
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.active(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

model = Model().to(device)
criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)

if __name__ == '__main__':
    epoch_list = []
    loss_list = []
    # Training Cycle
    for epoch in range(60):
        # Forward
        Los = 00
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            loss = loss.to(device)
            # print(epoch,i,loss.item())
            Los += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Los /= len(train_loader)
        print(epoch,Los)
        epoch_list.append(epoch)
        loss_list.append(Los)
    # data = testset[0]
    # inputs,labels = data
    # inputs = inputs.to(device)
    # # print(inputs)
    # torch.onnx.export(model,inputs,'.\model.onnx',export_params=True,opset_version=8,)
    Ans = np.empty((2,))
    for row in range(testset.len):
        data = testset[row]
        inputs,labels = data
        inputs = inputs.to(device)
        # print(inputs)
        y_pred = model(inputs).item()
        print(labels.item(),y_pred)
        if(y_pred > 0.445):
            A = [int(labels.item()),1]
        else:
            A = [int(labels.item()),0]
        Ans = np.row_stack((Ans,np.array(A,dtype = int)))
    Ans = Ans.astype(int)
    Ans = np.delete(Ans,0,0)
    # print(Ans)
    H = 'PassengerId,Survived'
    np.savetxt("code/data/titanic/ans.csv",Ans,fmt = '%d',delimiter = ',',header = H,comments='')
    # plt.plot(epoch_list, loss_list)
    # plt.ylabel('loss_list')
    # plt.xlabel('epoch_list')
    # plt.show()