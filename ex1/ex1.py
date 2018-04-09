# Machine Learning course by Anreww Ng
# Python version
# Solution by: Amin Ahmadi
# Date: Jan 12, 2018
############################################################
import numpy as np

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
    hypothesis parameters
    """
    m = Y.shape[0]
    # value of cost function is saved for plotting
    J_arr = np.zeros((num_iters), float)
    theta_arr = np.zeros((num_iters,2), float)
    
    for it in range(num_iters):
        error = np.dot(X,theta) - Y
        aux0 = theta[0] - (alpha/m) * np.sum(error*X[:,0])
        aux1 = theta[1] - (alpha/m) * np.sum(error*X[:,1])
        # update theta
        theta = np.array([aux0, aux1], float)

        # save data for plotting
        theta_arr[it] = theta
        J_arr[it] = J_cost(X,Y,theta)

    return theta, J_arr, theta_arr

############################################################
# 1 generate a identity matrix of size 5x5
A = np.eye(5, dtype=float)


# read data from file ex1data1.txt into array data
# 1st column: population
# 2nd column: profit
data = np.loadtxt('ex1data1.txt', dtype=float, delimiter=',')
X = data[:,0]; Y = data[:,1]    # feature and goal
print(X.shape)
m = X.shape[0]                     # number of training example

############################################################
# Implementation of Gradient Descent to determine hypothesis
############################################################
X = np.c_[np.ones(m), X] # add a column of one to X
theta = np.zeros(2)    # initialize fitting parameter
num_iters = 1500
alpha = 0.01
J_arr = np.zeros((num_iters), float)
theta_arr = np.zeros((num_iters,2), float)

print("Cost: ", J_cost(X,Y,theta))
theta, J_arr, theta_arr =  g_desc(X,Y, theta, alpha, num_iters)

print( "theta = ", theta)

##################################################
###                 plot data                  ###
##################################################
import matplotlib.pyplot as pl

fig = pl.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1)
ax.plot(data[:,0], data[:,1], '*k')
ax.plot(data[:,0], theta[0] + theta[1]*data[:,0], 'b-')
ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000a')
ax.set_xlim(4,)

ax1 = fig.add_subplot(1,2,2)
ax1.plot(np.linspace(0,1,num_iters), theta_arr[:,0] )
ax1.plot(np.linspace(0,1,num_iters),theta_arr[:,1])

# ax1.contour(J_arr, theta_arr[:,0], theta_arr[:,1])

pl.show()
