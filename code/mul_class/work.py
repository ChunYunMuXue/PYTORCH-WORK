import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
bath_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) #正态分布归一化
])

train_dataset = datasets.MNIST(root='./data/mnist',train=True,download=True,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=bath_size,shuffle=True)
test_dataset = datasets.MNIST(root='./data/mnist',train=False,download=True,transform = transform)
test_loader = DataLoader(test_dataset,batch_size=bath_size,shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()  
        self.line1 = torch.nn.Linear(784,512)
        self.line2 = torch.nn.Linear(512,256)
        self.line3 = torch.nn.Linear(256,128)
        self.line4 = torch.nn.Linear(128,64)
        self.line5 = torch.nn.Linear(64,10)
    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.line1(x))
        x = F.relu(self.line2(x))
        x = F.relu(self.line3(x))
        x = F.relu(self.line4(x))
        return self.line5(x) # 要接入到softmax层，最后一层线性输出到softmax
    
model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum = 0.5) # momentum 动量

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] Loss : %.3f' % (epoch + 1,batch_idx + 1,running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():#下列作用域内的代码不会再生成计算图
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test : ',(100 * correct / total),'%')

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

