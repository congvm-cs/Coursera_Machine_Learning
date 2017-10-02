"""Agenda
1 WarmUp Exercise
2 Linear Regression
3 Gradient Descent
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1 WarmUp
# Modify and return a 5x5 identity matrix 
warmUp = np.identity(5, dtype=int)

#2 Linear Regression
# Loading dataset
dataset = pd.read_csv('ex1data1.txt', header=None)
X = np.matrix(dataset.iloc[:, 0].values).T
y = np.matrix(dataset.iloc[:, 1].values).T

# Visualizing dataset
# plt.scatter(X, y, marker='o', c ='green')
# plt.show()

def computeCost(X, y, theta):
    #   L = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y
    m = len(X)
    L = (0.5/m)*np.sum(np.square(np.dot(X, theta) - y))
    return L

def cost_function(X, y, theta):
    loss = (0.5/len(X))*np.dot((np.dot(X, theta) - y).T, (np.dot(X, theta) - y))
    return loss

def gradient_descent(X, y, theta, iters, learning_rate):
    loss_history = []
    for i in xrange(iters):
        dJ = X.T * (np.dot(X, theta) - y)
        
        theta = theta - (1/len(X))*learning_rate*dJ
        
        loss_history.append(cost_function(X, y, theta))
    return theta, loss_history


def gradientDescent(X, y, iters, theta, alpha):
    #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    #   taking num_iters gradient steps with learning rate alpha
    m = len(y); # number of training examples
    J_history = []

    for iters in xrange(iters):
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        theta = theta - (alpha/m) * X.T *(np.dot(X, theta) - y)
        # ============================================================

        # Save the cost J in every iteration    
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

# Fitting dataset
iters = 1500
theta = np.zeros((2, 1))
alpha = 0.01 # learning rate

# adding bias
X_with_bias = np.concatenate((np.ones((len(X), 1)), X), axis=1)
loss = cost_function(X_with_bias, y, theta)

opt_theta, loss_history = gradientDescent(X_with_bias, y, iters, theta, alpha)
#print(opt_theta)
new_theta, loss_history1 = gradient_descent(X_with_bias, y, theta, iters, alpha)
'''
# Visualizing fitting line
def fitting_visualizing(X, y, theta):
    plt.scatter(X, y, marker='o', c ='green')
    xx = np.arange(min(X), max(X), 0.1)
    yy = theta[0] + theta[1]*xx
    plt.plot(xx, yy.T, c='red', linewidth=2)
    plt.show()

# fitting_visualizing(X, y, opt_theta)

def normalEquation(X, y):
    opt_theta = np.linalg.pinv(X.T*X)*X.T*y
    return opt_theta
opt_theta = normalEquation(X_with_bias, y)
print(opt_theta)
'''