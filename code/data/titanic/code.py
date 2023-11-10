import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 

class DiabetesDataset(Dataset):
    def __init__(self,path):
        X = np.empty((8,))
        Y = np.empty((1,))
        with open(path) as F:
            read = csv.DictReader(F)
            for Line in read:
                if('Survived' in Line.keys()):
                    Y = np.row_stack((Y,np.array([(1 - 0.01) if float(Line['Survived']) else 0.01])))
                else:
                    Y = np.row_stack((Y,np.array([float(1)])))
                del Line['Cabin']
                del Line['Name']
                del Line['Ticket']
                if('Survived' in Line.keys()):
                    del Line['Survived']
                Line['Sex'] = 1 if Line['Sex'] == 'male' else 0
                Line['Embarked'] = 1 if Line['Embarked'] == 'C' else 2 if Line['Embarked'] == 'S' else 3
                A = []
                for x in Line.keys():
                    if(Line[x] == ''):Line[x] = 0
                    A.append(float(Line[x]))
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

dataset = DiabetesDataset('code/data/titanic/train.csv')
testset = DiabetesDataset('code/data/titanic/test.csv')
train_loader = DataLoader(dataset = dataset,batch_size = 300,shuffle = True,num_workers = 2)

# print(x_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.active = torch.nn.ReLU() 
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.active(self.linear1(x))
        x = self.active(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model().to(device)
criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)

if __name__ == '__main__':
    epoch_list = []
    loss_list = []
    # Training Cycle
    for epoch in range(100):
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
    Ans = np.empty((2,))
    for row in range(testset.len):
        data = testset[row]
        inputs,labels = data
        inputs = inputs.to(device)
        # print(inputs)
        y_pred = model(inputs)
        optimizer.zero_grad()
        y_pred.backward()
        print(inputs[0],y_pred)
        if(y_pred.item() > 0.45):
            A = [int(inputs[0].item()),1]
        else:
            A = [int(inputs[0].item()),0]
        Ans = np.row_stack((Ans,np.array(A,dtype = int)))
    Ans = Ans.astype(int)
    Ans = np.delete(Ans,0,0)
    # print(Ans)
    np.savetxt("code/data/titanic/ans.csv",Ans,fmt = '%d',delimiter = ',')
    # plt.plot(epoch_list, loss_list)
    # plt.ylabel('loss_list')
    # plt.xlabel('epoch_list')
    # plt.show()