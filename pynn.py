import numpy as np
from random import *
import matplotlib.pyplot as plt
import time

'''
input data
'''
# xcoords, ycoords, targets
x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
xnum = len(x1)

# y for two-classification
# category A:[1,0]  B:[0,1]
y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])



def output(x, W, b):
    return np.matmul(W, x) + b


'''
different activation and derivative functions
'''

def chooseDerivativeFunction(activationName, i, n, A, W, B, x, alpha):
    if activationName is "sigmoid":
        return derivative_sigmoid(A[n-i-1])
    if activationName is "tanh":
        return derivative_tanh(A[n-i-1])
    if activationName is "arctan":
        if i == n-1:
            out = output(x, W[n-i-1], B[n-i-1])
        else:
            out = output(A[n-i-2], W[n-i-1], B[n-i-1])
        return derivative_arctan(out)
    if activationName is "relu":
        if i == n-1:
            out = output(x, W[n-i-1], B[n-i-1])
        else:
            out = output(A[n-i-2], W[n-i-1], B[n-i-1])
        return derivative_relu(out)
    if activationName is "prelu":
        if i == n-1:
            out = output(x, W[n-i-1], B[n-i-1])
        else:
            out = output(A[n-i-2], W[n-i-1], B[n-i-1])
        return derivative_prelu(out, alpha)
    if activationName is "identity":
        return 1

def chooseActivationFunc(activationName, x, W, b, alpha):
    if activationName is "sigmoid":
        return activate_sigmoid(x, W, b)
    if activationName is "tanh":
        return activate_tanh(x, W, b)
    if activationName is "arctan":
        return activate_arctan(x, W, b)
    if activationName is "relu":
        return activate_relu(x, W, b)
    if activationName is "prelu":
        return activate_prelu(x, W, b, alpha)
    if activationName is "identity":
        return activate_tanh(x, W, b)


# sigmoid
def activate_sigmoid(x, W, b):
    return 1.0 / np.add(1, np.exp(-output(x, W, b)))
# f'(x) = f(x)*(1-f(x))
def derivative_sigmoid(fx):
    return np.multiply(fx, (1 - fx))


# Tanh
def activate_tanh(x, W, b):
    return 2.0 / np.add(1, np.exp(-2 * output(x, W, b))) - 1
# f'(x) = 1-f(x)^2
def derivative_tanh(fx):
    return 1 - fx * fx


# ArcTan
def activate_arctan(x, W, b):
    return np.arctan(output(x, W, b))
# f'(x) = 1/(x^2+1)
def derivative_arctan(x):
    return  1.0 / (x*x + 1)


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

def activate_relu(x, W, b):
    x = output(x, W, b)
    m, n = x.shape
    # for n = 1 here
    y = np.zeros((m, n))
    for i in range(0, m):
        y[i][0] = activateone_relu(x[i][0])
    return y

def derivative_relu(x):
    m, n = x.shape
    # for n = 1 here
    y = np.zeros((m, n))
    for i in range(0, m):
        y[i][0] = derivativeone_relu(x[i][0])
    return y


# Parameteric rectified Linear Unit (PRelu)
def activateone_prelu(x, alpha):
    if x < 0:
        return alpha * x
    else:
        return x

def derivativeone_prelu(x, alpha):
    if x < 0:
        return alpha
    else:
        return 1

def activate_prelu(x, W, b, alpha):
    x = output(x, W, b)
    m, n = x.shape
    # for n = 1 here
    y = np.zeros((m, n))
    for i in range(0, m):
        y[i][0] = activateone_prelu(x[i][0], alpha)
    return y

def derivative_prelu(x, alpha):
    m, n = x.shape
    # for n = 1 here
    y = np.zeros((m, n))
    for i in range(0, m):
        y[i][0] = derivativeone_prelu(x[i][0], alpha)
    return y


# Identity
def activate_identity(x, W, b):
    return output(x, W, b)


# initialize weights and biases
# Nlayer: the number of layer
# NneuronList: the numbers of neurons for each layer
def initialize_weights(Nseed, Nlayer, NneuronList):
    W = []
    for i in range(1, Nlayer):
        np.random.seed(Nseed + i*100)
        w = 0.5 * np.random.randn(NneuronList[i], NneuronList[i - 1])
        W.append(w)
    return W


def initialize_bias(Nseed, Nlayer, NneuronList):
    B = []
    for i in range(1, Nlayer):
        np.random.seed(Nseed + i*100)
        b = 0.5 * np.random.randn(NneuronList[i], 1)
        B.append(b)
    return B


def forward(activationName, x, W, B, alpha):
    A = []
    for i in range(0, len(W)):
        if i == 0:
            a = chooseActivationFunc(activationName, x, W[i], B[i], alpha)
        else:
            a = chooseActivationFunc(activationName, A[i - 1], W[i], B[i], alpha)
        A.append(a)
    return A


def backward(activationName, A, W, B, k, x, alpha):
    Delta = []
    # Backward pass
    n = len(A)
    for i in range(0, n):
        if i == 0:
            delta = np.multiply(chooseDerivativeFunction(activationName, i, n, A, W, B, x, alpha),
                                A[n - i - 1] - y[:, k].reshape(2, 1))
        else:
            delta = np.multiply(chooseDerivativeFunction(activationName, i, n, A, W, B, x, alpha),
                                np.transpose(W[n - i]).dot(Delta[i - 1]))
        Delta.append(delta)
    return Delta


def gradient(Delta, A, W, B, x, eta):
    n = len(A)
    ans = []
    for i in range(0, n):
        if i == 0:
            W[i] = W[i] - eta * Delta[n-i-1] * np.transpose(x)
        else:
            W[i] = W[i] - eta * Delta[n-i-1] * np.transpose(A[i-1])
        B[i] = B[i] - eta * Delta[n-i-1]

    ans.append(W)  # ans[0] = W
    ans.append(B)  # ans[1] = B
    return ans


# calculate the cost
def cost(activationName, W, B, alpha):
    costvec = np.zeros((xnum, 1))
    for i in range(0, xnum):
        x = np.array([x1[i], x2[i]]).reshape(2, 1)
        A = forward(activationName, x, W, B, alpha)
        costvec[i] = np.linalg.norm(y[:, i].reshape(2, 1) - A[len(W)-1])
    return np.linalg.norm(costvec) ** 2


# Forward and Back propagate
def main_nn(Nlayer, NneuronList, eta, Niter, Nseed, activationName, threshold, alpha=0.5, plotFlag=True,
            timeFlag=True):
    W = initialize_weights(Nseed, Nlayer, NneuronList)
    B = initialize_bias(Nseed, Nlayer, NneuronList)

    savecost = np.zeros((Niter, 1))
    np.random.seed(Nseed)

    for counter in range(0, Niter):
        k = randint(0, xnum-1)
        # Pick a training point at random
        x = np.array([x1[k], x2[k]]).reshape(2, 1)

        # Forward pass
        A = forward(activationName, x, W, B, alpha)

        # Backward pass
        Delta = backward(activationName, A, W, B, k, x, alpha)  # caution: delta from n to 2

        # Gradient step
        ans = gradient(Delta, A, W, B, x, eta)
        W = ans[0]
        B = ans[1]

        # Monitor progress
        newcost = cost(activationName, W, B, alpha)
        print(newcost)  # display cost to screen
        savecost[counter] = newcost
        actualIter = counter
        if newcost < threshold:
            break

    if timeFlag:
        # stop measuring the time
        end = time.time()
        elapsed = end - start

    if plotFlag:
        originalPlotName = "xy.png"
        costPlotName = "2232_cost_" + str(Nlayer) + "layer_" + activationName + "_" + str(eta) \
                       + "eta_" + str("{:.0e}".format(Niter)) + "iteration_" + str(threshold) + "threshold.png"
        classRegionPlotName = "2232_classRegion_" +  str(Nlayer) + "layer_" + activationName + "_" + str(eta) \
                              + "eta_" + str("{:.0e}".format(Niter)) + "iteration_" + str(threshold) + "threshold.png"
        plot_originalTargets(x1, x2, 5, 5, originalPlotName)
        # print(actualIter)
        plot_cost(eta, savecost, elapsed, costPlotName, actualIter, threshold)
        plot_classificationRegion(activationName, classRegionPlotName, W, B, alpha)


# draw the curve of the cost function
def plot_cost(eta, savecost, elapsed, savefigname, actualIter, threshold):
    fig2 = plt.figure()
    iter = int(actualIter / 100)
    iteration = list(range(0, actualIter + 1000, iter))
    savecost_list = savecost[0: actualIter + 1000: iter]
    plt.semilogy(iteration, savecost_list[0:actualIter + 1000])
    plt.text(actualIter, threshold, str(actualIter))
    # plt.text(actualIter, threshold, '({}, {})'.format(str(threshold), str(actualIter)))
    plt.text(actualIter, threshold, actualIter)
    plt.plot(actualIter, threshold, "ro")
    plt.title(
        "To " + str(threshold) + " loss : " + str(eta) + "eta, " + str(actualIter) + " iterations  in " + str(
            round(elapsed, 4)) + " seconds")
    plt.xlabel('Iteration Number')
    plt.ylabel('Value of cost function')
    plt.savefig(savefigname)
    plt.close(fig2)


# display shaded and unshaded regions
def plot_classificationRegion(activationName, savefigname, W, B, alpha, N=500):
    Dx = 1 / N
    Dy = 1 / N
    xvals = np.arange(0, 1.002, Dx)
    yvals = np.arange(0, 1.002, Dy)
    Aval = np.zeros((N + 1, N + 1))
    Bval = np.zeros((N + 1, N + 1))
    n = len(W)
    for k1 in range(0, N + 1):
        xk = xvals[k1]
        for k2 in range(0, N + 1):
            yk = yvals[k2]
            xy = np.array([xk, yk]).reshape(2, 1)
            A = forward(activationName, xy, W, B, alpha)
            Aval[k2][k1] = A[n-1][0]
            Bval[k2][k1] = A[n-1][1]
    X, Y = np.meshgrid(xvals, yvals)

    fig3 = plt.figure()
    Mval = Aval > Bval
    mycmap2 = plt.get_cmap('Greys')
    cf2 = plt.contourf(X, Y, Mval, cmap=mycmap2)
    plt.plot(x1[0:5], x2[0:5], 'ro')
    plt.plot(x1[5:10], x2[5:10], 'b+')
    # plt.plot(0.1,0.1,'y^')
    plt.FontSize = 16
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(savefigname)
    plt.close(fig3)


# draw the original targets
def plot_originalTargets(x1, x2, class1num, class2num, savefigname):
    fig1 = plt.figure()
    plt.plot(x1[0:class1num], x2[0:class1num], 'ro')
    plt.plot(x1[class1num:class1num + class2num], x2[class1num:class1num + class2num], 'b+')
    plt.title("original targets")
    plt.savefig(savefigname)
    plt.close(fig1)


if __name__ == "__main__":
    # change the following parameters to create various neural networks
    eta = 0.01    # learning rate
    Niter = int(1e9)    # epoch
    seed = 0    # seed for np.random
    activationName = "sigmoid"  # activation function name, must be lowercase.
    # Choices: ["sigmoid","tanh", "arctan", "relu", "prelu", "identity"]
    # alpha = 0.5   # alpha for prelu, default is 0.5
    threshold = 1e-3    # when the cost is lower than threshold, the training will stop
    Nlayer = 4  # number of layers
    NneuronList = [2,2,3,2]     # number of neurons ofr each layer
    # two flag must be same because cost plot need time
    plotFlag = True
    timeFlag = True

    if timeFlag:
        start = time.time()
        np.random.seed(0)

    main_nn(Nlayer, NneuronList, eta, Niter, seed, activationName, threshold, plotFlag, timeFlag)


