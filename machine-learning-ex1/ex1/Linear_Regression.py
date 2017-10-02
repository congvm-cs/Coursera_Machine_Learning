import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data
dataset = pd.read_csv('ex1data1.txt', header=None)
X = np.matrix(dataset.iloc[:, 0].values).T
y = np.matrix(dataset.iloc[:, 1].values).T

# Visualize
plt.scatter(X, y, c='red', marker='o')

# Adding bias
bias = np.ones((X.shape))
X = np.concatenate((bias, X), axis=1)

# Random theta
# theta = np.random.random((2, 1))
theta = np.ones((2, 1))


def loss_function(X, y, theta):
    print('Computing loss')
    # loss = (0.5 / len(X)) * np.dot((np.dot(X, theta) - y).T, (np.dot(X, theta) - y))
    loss = (0.5 / len(X)) * (X*theta - y).T * (X*theta - y)
    return loss


def gradient_descent(X, y, theta, iterations, learning_rate):
    loss_history = []
    for i in range(iterations):
        dJ = (np.dot(X.T, (np.dot(X, theta) - y)))
        theta = theta - (1 / len(X)) * learning_rate * dJ
        loss_history.append(loss_function(X, y, theta))
    return theta, loss_history


loss = loss_function(X=X, y=y, theta=theta)
