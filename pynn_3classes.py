import numpy as np
from random import *
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn import datasets
# iris = datasets.load_iris()
# X_train, X_test, y_train, y_test = \
#     train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
'''
this is for 3-class classification
'''

# xcoords, ycoords, targets
x1 = np.array([0.1,0.3,0.1,0.4,0.6,0.9,0.4,0.7,0.6,0.5])
x2 = np.array([0.1,0.4,0.5,0.2,0.3,0.2,0.4,0.6,0.9,0.6])
# y_3 for three-classification
# category A:[0]  B:[1] C:[2]
y_3 = np.array([0,0,0,0,1,1,1,2,2,2])


# draw the orginal targets
fig1 = plt.figure()
plt.plot(x1[0:4],x2[0:4],'ro')
plt.plot(x1[4:7],x2[4:7],'b+')
plt.plot(x1[7:10],x2[7:10],'g^')
plt.title("original targets")
plt.savefig('pic_xy_3classes.png')
plt.close(fig1)

def output(x, W, b):
    return np.matmul(W,x)+b

# Tanh
# def activate(x,W,b):
#     return 2.0 / np.add(1, np.exp(-2*(np.matmul(W,x)+b))) - 1
# Relu
def activateone_relu(x):
    if x < 0:
        return 0
    else:
        return x

def derivativeone_relu(x):
    if x < 0:
        return 0
    else:
        return 1

def activate(x, W, b):
    x = np.matmul(W, x) + b
    m,n = x.shape
    # for n = 1 here
    y = np.zeros((m,n))
    for i in range(0,m):
        y[i][0] = activateone_relu(x[i][0])
    return y

def derivative_relu(x):
    m, n = x.shape
    # for n = 1 here
    y = np.zeros((m, n))
    for i in range(0, m):
        y[i][0] = derivativeone_relu(x[i][0])
    return y

# calculate the cost
def cost(W2, W3, W4, b2, b3, b4):
    costvec = np.zeros((10, 1))
    for i in range(0, 10):
        x = np.array([x1[i], x2[i]]).reshape(2, 1)
        a2 = activate(x, W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        costvec[i] = np.linalg.norm(y_3[i] - a4)
    return np.linalg.norm(costvec) ** 2

# start measuring the time
start = time.time()

# initialize weights and biases
# worse if using rand which generated numbers are all positive
np.random.seed(0)

# 2 - 9 - 9 - 1
W2 = 0.5 * np.random.randn(9, 2)
W3 = 0.5 * np.random.randn(9, 9)
W4 = 0.5 * np.random.randn(2, 9)
b2 = 0.5 * np.random.randn(9, 1)
b3 = 0.5 * np.random.randn(9, 1)
b4 = 0.5 * np.random.randn(1, 1)

# Forward and Back propagate
# Pick a training point at random
eta = 0.05
Niter = int(1e5)
savecost = np.zeros((Niter,1))
seed(0)

for counter in range (0,Niter):
    k = randint(0,9)
    x = np.array([x1[k], x2[k]]).reshape(2,1)
    # Forward pass
    a2 = activate(x,W2,b2)
    a3 = activate(a2,W3,b3)
    a4 = activate(a3,W4,b4)
    # Tanh
    # delta4 = np.multiply(1 - a4 * a4, (a4 - y_3[k]))
    # delta3 = np.multiply(1 - a3 * a3, np.transpose(W4).dot(delta4))
    # delta2 = np.multiply(1 - a2 * a2, np.transpose(W3).dot(delta3))
    # Relu
    # f'(x) = 0 for x<0; otherwise = 1
    output2 = output(x, W2, b2)
    output3 = output(a2, W3, b3)
    output4 = output(a3, W4, b4)
    delta4 = np.multiply(derivative_relu(output4), (a4 - y_3[k]))
    delta3 = np.multiply(derivative_relu(output3), np.transpose(W4).dot(delta4))
    delta2 = np.multiply(derivative_relu(output2), np.transpose(W3).dot(delta3))
    # Gradient step
    W2 = W2 - eta * delta2 * np.transpose(x)
    W3 = W3 - eta * delta3 * np.transpose(a2)
    W4 = W4 - eta * delta4 * np.transpose(a3)
    b2 = b2 - eta * delta2
    b3 = b3 - eta * delta3
    b4 = b4 - eta * delta4
    # Monitor progress
    newcost = cost(W2, W3, W4, b2, b3, b4)
    print(newcost)  # display cost to screen
    savecost[counter] = newcost

# stop measuring the time
end = time.time()
elapsed = end - start


# draw the curve of the cost function
#
fig2 = plt.figure()
iter = int(1e3)  # 1e4
iteration = list(range(0, Niter, iter))
savecost_list = savecost[0: Niter: iter]
plt.semilogy(iteration, savecost_list)
plt.title(str(eta) + "eta, " + str("{:.2e}".format(Niter)) + " iterations  in " + str(round(elapsed, 4)) + " seconds")
plt.xlabel('Iteration Number')
plt.ylabel('Value of cost function')
plt.savefig('pic_cost_relu_3classes_2992_0.05eta_1e5_test.png')
plt.close(fig2)


# display shaded and unshaded regions
N = 500
Dx = 1/N
Dy = 1/N
xvals = np.arange(0,1.002,Dx)
yvals = np.arange(0,1.002,Dy)
Aval = np.zeros((N+1,N+1))
Bval = np.zeros((N+1,N+1))
Cval = np.zeros((N+1,N+1))
for k1 in range(0,N+1):
    xk = xvals[k1]
    for k2 in range(0,N+1):
        yk = yvals[k2]
        xy = np.array([xk,yk]).reshape(2,1)
        a2 = activate(xy,W2,b2)
        a3 = activate(a2,W3,b3)
        # 4 layer
        a4 = activate(a3,W4,b4)
        Acost = np.abs(a4[0])
        Bcost = np.abs(a4[0]-1)
        Ccost = np.abs(a4[0]-2)
        Aval[k2][k1] = 0
        Bval[k2][k1] = 0
        Cval[k2][k1] = 0
        if Ccost < Acost and Ccost < Bcost:
            Cval[k2][k1] = 1
        if Bcost < Acost and Bcost < Ccost:
            Bval[k2][k1] = 1
        if Acost < Bcost and Bcost < Ccost:
            Aval[k2][k1] = 1
X,Y = np.meshgrid(xvals,yvals)

fig3 = plt.figure()
Mval = (Aval>Bval) & (Aval>Cval)
Vval = (Bval>Aval) & (Bval>Cval)
mycmap2 = plt.get_cmap('Greys')
cf2 = plt.contourf(X,Y,Mval,cmap=mycmap2)
cf2 = plt.contourf(X,Y,Vval,cmap=mycmap2)
plt.plot(x1[0:4],x2[0:4],'ro')
plt.plot(x1[4:7],x2[4:7],'b+')
plt.plot(x1[7:10],x2[7:10],'g^')
plt.FontSize = 16
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('pic_bdy_bp_relu_3classes_2992_0.05eta_1e5_test.png')
plt.close(fig3)





# draw the activation functions

# x = np.arange(-4,4,0.1)
# y = np.arctan(x)  # arctan
# y1 = 1.0/(1+np.exp(-x))     # sigmoid
# y2 = 2.0/(1+np.exp(-2*x)) -1    # tanh
# y3 = x  # identity
#
# #relu
# def gety4(x):
#     m = len(x)
#     y = np.zeros((m,1))
#     for i in range(0, m):
#         if x[i] < 0:
#             y[i] = 0
#         else:
#             y[i] = x[i]
#     return y
#
# # prelu
# def gety5(x):
#     alpha = 0.1
#     m = len(x)
#     y = np.zeros((m, 1))
#     for i in range(0, m):
#         if x[i] < 0:
#             y[i] = alpha*x[i]
#         else:
#             y[i] = x[i]
#     return y
#
# plt.title("activation functions")
# plt.plot(x,y, label = "arctan")
# plt.plot(x,y1, label = "sigmoid")
# plt.plot(x,y2, label = "tanh")
# plt.plot(x,y3, label = "identity")
# plt.plot(x,gety4(x), label = "relu")
# plt.plot(x,gety5(x), label = "prelu")
# plt.legend()
#
# plt.show()