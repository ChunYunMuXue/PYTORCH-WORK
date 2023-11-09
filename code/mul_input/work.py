import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

xy = np.loadtxt('code\mul_input\diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])
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
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)

epoch_list = []
loss_list = []
# Training Cycle
for epoch in range(1000):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    print(epoch, loss.item())
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss_list')
plt.xlabel('epoch_list')
plt.show()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# xy = np.loadtxt('code\mul_input\diabetes.csv.gz', delimiter=',', dtype=np.float32)
# x_data = torch.from_numpy(xy[ : ,  :-1])    # 第一个:的前后参数缺省，表示读取所有行 第二个:指从开始列到倒数第二列
# y_data = torch.from_numpy(xy[ : , [-1]])    # [-1]使向量转换为矩阵  from_numpy可以生成tensor
# # test = np.loadtxt('test.csv.gz', delimiter='', dtype=np.float32)


# # print(y_data)

# # Define Model
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear1 = torch.nn.Linear(8, 6)    # 表示8维->6维 空间变换
#         self.linear2 = torch.nn.Linear(6, 4)
#         self.linear3 = torch.nn.Linear(4, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.sigmoid(self.linear1(x))
#         x = self.sigmoid(self.linear2(x))
#         x = self.sigmoid(self.linear3(x))
#         return x

# model = Model()

# # Construct Loss and Optimizer
# criterion = torch.nn.BCELoss(reduction = 'mean')
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# epoch_list = []
# loss_list = []
# # Training Cycle
# for epoch in range(100):
#     # Forward
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     epoch_list.append(epoch)
#     loss_list.append(loss.item())
#     print(epoch, loss.item())
#     # Backward
#     optimizer.zero_grad()
#     loss.backward()
#     # Update
#     optimizer.step()

# plt.plot(epoch_list, loss_list)
# plt.ylabel('loss_list')
# plt.xlabel('epoch_list')
# plt.show()

