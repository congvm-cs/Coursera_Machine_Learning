import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
dataset = pd.read_csv('ex1data1.txt', header=None)
X = np.matrix(dataset.iloc[:, 0]).T
y = np.matrix(dataset.iloc[:, 1]).T

X_with_bias = np.concatenate((np.ones(X.shape), X), axis=1)
theta = np.zeros((2, 1))


def loss_function(X, y, theta):
    m = len(X)
    return (0.5/m) * (X*theta - y).T * (X*theta - y)


def gradient_descent(X, y, theta, epochs, learning_rate):
    m = len(X)
    loss_history = []
    for i in range(epochs):
        dJ = (1/m) * X.T * (X*theta - y)
        theta = theta - learning_rate*dJ
        loss_history.append(loss_function(X, y, theta))
    return theta, loss_history


def visualize(X, y, theta):
    plt.plot(X, y, 'ro')
    xx = range(int(X.min()), int(X.max()))
    yy = theta[0, 0] + xx*theta[1, 0]
    plt.plot(xx, yy, 'b-')
    plt.show()
loss = loss_function(X_with_bias, y, theta)

epochs = 10000
learning_rate = 0.01
theta, loss_history = gradient_descent(X_with_bias, y, theta, epochs, learning_rate)

visualize(X, y, theta)