
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
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
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size = 5)
        self.conv3 = torch.nn.Conv2d(20,30,kernel_size = 3)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(30,10)
    
    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum = 0.5) # momentum 动量

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] Loss : %.3f' % (epoch + 1,batch_idx + 1,running_loss / 300))
            running_loss = 0.0

E = []
A = []

def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():#下列作用域内的代码不会再生成计算图
        for data in test_loader:
            images,labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test : ',(100 * correct / total),'%')
    A.append(correct / total)
    E.append(epoch)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)
    F = plt.figure()    
    plt.plot(E,A,'royalblue');  
    plt.ylabel('Accarcy')
    plt.xlabel('Epoch')
    plt.show()