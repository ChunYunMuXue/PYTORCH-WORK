#eg:trans hello -> ohlol

import torch 

idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

datas = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

dataset = [datas[x] for x in x_data]

batch_size = 1
input_size = 4
hidden_size = 4

inputs = torch.Tensor(dataset).view(-1,batch_size,input_size)
labels = torch.LongTensor(y_data).view(-1,1)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size = self.input_size,hidden_size = self.hidden_size)

    def forward(self,input,hidden):
        hidden = self.rnncell(input,hidden)
        return hidden
    
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

net = Model(input_size,hidden_size,batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0 
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print("Predicted string : ",end = '')
    for input,label in zip(inputs,labels):
        hidden = net(input,hidden)
        print("Shape:",hidden.shape)
        loss += criterion(hidden,label)
        _,idx = hidden.max(dim = 1)
        print(idx2char[idx.item()],end = '')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))