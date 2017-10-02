import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ex1data2.txt', header=None)
X = np.matrix(dataset.iloc[:, :2])
y = np.matrix(dataset.iloc[:, 2]).T

# Add bias
X_with_bias = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# theta = np.random.random((X_with_bias.shape[1], 1))
theta = np.zeros((X_with_bias.shape[1], 1))


def loss_function(X, y, theta):
    m = float(len(X))
    loss = (0.5 / m) * (X * theta - y).T * (X * theta - y)
    return loss


def gradient_descent(X, y, theta, iterations, learning_rate):
    loss_history = []
    m = float(len(X))
    for i in range(iterations):
        dJ = X.T * (X * theta - y)
        theta = theta - (learning_rate / m) * dJ
        loss_history.append(loss_function(X, y, theta))
    return theta, loss_history

iterations = 1000
learning_rate = 0.1
loss = loss_function(X_with_bias, y, theta)
theta, loss_history = gradient_descent(X_with_bias, y, theta, iterations, learning_rate)

yy = np.linspace(np.min(loss_history), np.max(loss_history), 25)
plt.plot(loss_history, yy , 'r-')
plt.show()