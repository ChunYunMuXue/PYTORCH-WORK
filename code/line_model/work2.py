import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print(0.7*15+0.7*20+0.9*10+0.9*10+0*10+0.6*15+0.8*20)

x_data = [1.0,2.0,3.0]
y_data = [4.0,6.0,8.0]
W = []
B = []
Z = []


def value(x,w,b):return x * w + b

def loss(w,b):
    sum = 0
    for t in range(3):
        y = value(x_data[t],w,b)
        sum += (y - y_data[t]) * (y - y_data[t])
    return sum / 3

X = np.arange(0.0,4.1,0.1)
Y = np.arange(0.0,4.1,0.1)
X,Y = np.meshgrid(X,Y)
Z = X.copy()
for i in range(len(X)):
    for j in range(len(X[0])):
        Z[i][j] = loss(X[i][j],Y[i][j])
# print(X)
# print(Z)

fig = plt.figure()
ax3 = plt.axes(projection='3d')
plt.title('LOSS FUNCTION')
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
plt.show()
