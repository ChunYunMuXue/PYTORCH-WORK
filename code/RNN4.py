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
num_layers = 1
seq_len = 5

inputs = torch.Tensor(dataset).view(seq_len,batch_size,input_size)
labels = torch.LongTensor(y_data).view(-1,1)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size = self.input_size,hidden_size = self.hidden_size,num_layers = num_layers)

    def forward(self,input):
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out,_ = self.rnn(inputs,hidden)
        return out.view(-1,self.hidden_size)
    
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

net = Model(input_size,hidden_size,batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0 
    optimizer.zero_grad()
    hidden = net.init_hidden()
    hiddens = net(inputs) 
    for hidden,label in zip(hiddens,labels):
        print("Shape :",hidden.shape)
        hidden = hidden.view(-1,hidden_size)
        loss += criterion(hidden,label)
        _,idx = hidden.max(dim = 1)
        print(idx2char[idx.item()],end = '')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))