import numpy as np
from random import *
import matplotlib.pyplot as plt
import time


# xcoords, ycoords, targets
x1 = np.array([0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7])
x2 = np.array([0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6])
y = np.array([[1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1]])

# draw the orginal targets
fig1 = plt.figure()
plt.plot(x1[0:5],x2[0:5],'ro')
plt.plot(x1[5:10],x2[5:10],'b+')
plt.title("original targets")
plt.savefig('pic_xy.png')
plt.close(fig1)

def activate(x,W,b):
    return 1.0 / np.add(1, np.exp(-(np.matmul(W,x)+b)))

def cost(W2,W3,W4,b2,b3,b4):
     costvec = np.zeros((10,1))
     for i in range (0,10):
         x = np.array([x1[i],x2[i]]).reshape(2,1)
         a2 = activate(x,W2,b2)
         a3 = activate(a2,W3,b3)
         a4 = activate(a3,W4,b4)
         costvec[i]= np.linalg.norm(y[:,i].reshape(2,1) - a4)
     return np.linalg.norm(costvec)**2

# start measuring the time
start = time.time()


# initialize weights and biases
# worse if using rand which generated numbers are all positive
np.random.seed(0)
W2 = 0.5*np.random.randn(2,2)
W3 = 0.5*np.random.randn(3,2)
W4 = 0.5*np.random.randn(2,3)
b2 = 0.5*np.random.randn(2,1)
b3 = 0.5*np.random.randn(3,1)
b4 = 0.5*np.random.randn(2,1)
end1 = time.time()

#  this is for time measurement
elapsed2 = 0
elapsed3 = 0
elapsed4 = 0
elapsed5 = 0


# Forward and Back propagate
# Pick a training point at random
eta = 1
Niter = int(1e6)
savecost = np.zeros((Niter,1))
seed(5000)
for counter in range (0,Niter):
    k = randint(0,9)
    x = np.array([x1[k], x2[k]]).reshape(2,1)
    # Forward pass
    a2 = activate(x,W2,b2)
    a3 = activate(a2,W3,b3)
    a4 = activate(a3,W4,b4)
    # this is for time measurement
    end2 = time.time()
    if counter == 0:
        elapsed2 = end2-end1
    else:
        elapsed2 += end2-end5

    # Backward pass
    delta4 = np.multiply(np.multiply(a4, (1-a4)), (a4-y[:,k].reshape(2,1)))
    delta3 = np.multiply(np.multiply(a3, (1 - a3)), np.transpose(W4).dot(delta4))
    delta2 = np.multiply(np.multiply(a2, (1 - a2)), np.transpose(W3).dot(delta3))
    # this is for time measurement
    end3 = time.time()
    elapsed3 += end3-end2

    # Gradient step
    W2 = W2 - eta*delta2*np.transpose(x)
    W3 = W3 - eta*delta3*np.transpose(a2)
    W4 = W4 - eta*delta4*np.transpose(a3)
    b2 = b2 - eta*delta2
    b3 = b3 - eta*delta3
    b4 = b4 - eta*delta4
    # this is for time measurement
    end4 = time.time()
    elapsed4 += end4-end3

    # Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4)
    print (newcost)     # display cost to screen
    savecost[counter] = newcost
    # this is for time measurement
    end5 = time.time()
    elapsed5 += end5-end4

# stop measuring the time
end = time.time()
elapsed = end - start


# draw the curve of the cost function
fig2 = plt.figure()
iter = int(1e4)  # 1e4
# iter = int(Niter/100)  # 1e4
iteration = list(range(0, Niter, iter))
savecost_list = savecost[0: Niter: iter]
plt.semilogy(iteration, savecost_list)
plt.title(str(eta) + "eta, " + str("{:.2e}".format(Niter)) + " iterations  in " + str(round(elapsed, 4)) + " seconds")
plt.xlabel('Iteration Number')
plt.ylabel('Value of cost function')
plt.savefig('pic_cost_1eta_1e6.png')
plt.close(fig2)


# display shaded and unshaded regions
N = 500
Dx = 1/N
Dy = 1/N
xvals = np.arange(0,1.002,Dx)
yvals = np.arange(0,1.002,Dy)
Aval = np.zeros((N+1,N+1))
Bval = np.zeros((N+1,N+1))
for k1 in range(0,N+1):
    xk = xvals[k1]
    for k2 in range(0,N+1):
        yk = yvals[k2]
        xy = np.array([xk,yk]).reshape(2,1)
        a2 = activate(xy,W2,b2)
        a3 = activate(a2,W3,b3)
        a4 = activate(a3,W4,b4)
        Aval[k2][k1] = a4[0]
        Bval[k2][k1] = a4[1]
X,Y = np.meshgrid(xvals,yvals)

fig3 = plt.figure()
print(Aval)
print(Bval)
Mval = Aval>Bval
mycmap2 = plt.get_cmap('Greys')
cf2 = plt.contourf(X,Y,Mval,cmap=mycmap2)
plt.plot(x1[0:5],x2[0:5],'ro')
plt.plot(x1[5:10],x2[5:10],'b+')
plt.FontSize = 16
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('pic_bdy_bp_1eta_1e6.png')
plt.close(fig3)

print("initialize weights and biases: %f " % (end1-start))
print("front propagation: %f " % elapsed2)
print("back propagation: %f " % elapsed3)
print("gradient: %f " % elapsed4)
print("cost: %f " % elapsed5)
print(elapsed)

