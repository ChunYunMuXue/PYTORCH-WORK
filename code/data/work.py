import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,path):
        xy = np.loadtxt(path,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

dataset = DiabetesDataset('code\mul_input\diabetes.csv.gz')
train_loader = DataLoader(dataset = dataset,batch_size = 32,shuffle = True,num_workers = 2)

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

model = Model()
criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)

if __name__ == '__main__':
    epoch_list = []
    loss_list = []
    # Training Cycle
    for epoch in range(100):
        # Forward
        Los = 0
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            # print(epoch,i,loss.item())
            Los += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Los /= len(train_loader)
        print(epoch,Los)
        epoch_list.append(epoch)
        loss_list.append(Los)

    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss_list')
    plt.xlabel('epoch_list')
    plt.show()
