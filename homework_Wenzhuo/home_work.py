import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt5
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
import matplotlib.pyplot as plt
device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 

import math

bath_size = 20
input_size = 6

class DiabetesDataset(Dataset):
    def __init__(self,path):
        X = np.empty((input_size,))
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
                # print(Line['level'])
                Y = np.row_stack((Y,np.array(int(Line['level'] - 1))))
                del Line['num']
                A = []
                for x in Line.keys():A.append(float(Line[x]))
                X = np.row_stack((X,np.array(A[:-1],dtype = 'double')))
        X = np.delete(X,0,0)
        Y = np.delete(Y,0,0)       
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y).long()
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len


custom_dataset = DiabetesDataset('data.csv')

train_size = int(len(custom_dataset) * 0.7)
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset,batch_size = bath_size,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = bath_size,shuffle = False)

#_______________________________deal_with_dataset_______________________________

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()  
        self.line1 = torch.nn.Linear(input_size,10)
        self.line2 = torch.nn.Linear(10,5)
        self.line3 = torch.nn.Linear(5,7)
    def forward(self,x):
        x = x.view(-1,input_size)
        x = F.relu(self.line1(x))
        x = F.relu(self.line2(x))
        return self.line3(x) # 要接入到softmax层，最后一层线性输出到softmax
    
#_____________________________Make_Net___________________________________

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.00948)

LOSS_DATA = []
AC_RATE = []
X_DATA = []

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()
        target = target.reshape(target.size()[0])
        outputs = model(inputs)
        # print(outputs,target)
        # print(outputs.size(),target.size())
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step() 
        running_loss += loss.item()
        # if batch_idx % 2 == 0:
        #     print('[%d, %5d] Loss : %.3f' % (epoch + 1,batch_idx + 1,running_loss / 2))
        #     running_loss = 0.0
    LOSS_DATA.append(running_loss)
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            test,labels = data
            outputs = model(test)
            labels = labels.reshape(labels.size()[0])
            _,predicted = torch.max(outputs.data,dim = 1)
            total += labels.size(0)
            # print(predicted,labels)
            correct += (predicted == labels).sum().item()
    AC_RATE.append((100 * correct / total))        
    print('Accuracy on test : ',(100 * correct / total),'%')

if __name__ == '__main__':
    for epoch in range(1000):
        train(epoch)
        test()
        X_DATA.append(epoch)
    P = plt.figure(figsize = (100,80))
    ax1 = P.subplots()
    ax2 = ax1.twinx()
    ax1.plot(X_DATA,LOSS_DATA,'g-')
    ax2.plot(X_DATA,AC_RATE,'b--') 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    plt.show()