# Machine Learning course by Anreww Ng
# Python version
# Solution by: Amin Ahmadi
# Date: Jan 12, 2018
############################################################
import numpy as np
import matplotlib.pyplot as pl

############################################################

def J_cost(X,Y, theta):
    """ compute the cost function for given 
    features X and goals Y, theta is the fitting paramater
    """
    m = Y.shape[0]              # number of training data
    
    aux1 = np.dot(X,theta)
    aux2 = (Y-aux1)*(Y-aux1)
    J = (1./(2.*m)) * np.sum(aux2)
    
    return J
############################################################
def g_desc(X,Y, theta, alpha, num_iters):
    """ This function uses gradient descent to fit the 
    hypothesis parameters"""
    m = Y.shape[0]
    J_arr = np.zeros((num_iters), float)
    
    for it in range(num_iters):
        error = np.dot(X,theta) - Y
        aux0 = theta[0] - (alpha/m) * np.sum(error*X[:,0])
        aux1 = theta[1] - (alpha/m) * np.sum(error*X[:,1])
        # update theta
        theta = np.array([aux0, aux1], float)

        J_arr[it] = J_cost(X,Y,theta)

    return theta

############################################################
# 1 generate a identity matrix of size 5x5
A = np.eye(5, dtype=float)


# read data from file ex1data1.txt into array data
# 1st column: population
# 2nd column: profit
data = np.loadtxt('ex1data1.txt', dtype=float, delimiter=',')
X = data[:,0]; Y = data[:,1]    # feature and goal
print X.shape
m = X.shape[0]                     # number of training example

# plot data 
pl.plot(data[:,0], data[:,1], '*k')
pl.xlabel('Population of City in 10,000s')
pl.ylabel('Profit in $10,000a')
pl.xlim(4,)
pl.show()

############################################################
# Implementation of Gradient Descent to determine hypothesis
############################################################
X = np.c_[np.ones(m), X] # add a column of one to X
theta = np.zeros(2)    # initialize fitting parameter
num_iters = 1500
alpha = 0.01

print "Cost: ", J_cost(X,Y,theta)
print g_desc(X,Y, theta, alpha, num_iters)
