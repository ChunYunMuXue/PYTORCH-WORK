import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = []

for w in np.arange(0.0,4.1,0.1):
    print("w = ",w)
    sum = 0
    for x_val,y_val in zip(x_data,y_data):
        y_pv = forward(x_val)
        sum += loss(x_val,y_val)
        print('\t',x_val,y_val,y_pv,loss(x_val,y_val))
    print("MSE = ",sum / 3)
    w_list.append(w)
    mse_list.append(sum / 3)
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()