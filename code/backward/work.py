import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w # 会强制转换x为tensor，并自动创建计算图，返回值也是tensor

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)",4,forward(4).item()) # item,tensor 容器取值

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y) #一个tensor
        l.backward()
        print('\tgrad:',x,y,w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_() # 反向传播不会把grad清零
    print("progress:",epoch,l.item())
print("predict (after training)",4,forward(4).item()) # item,tensor 容器取值

