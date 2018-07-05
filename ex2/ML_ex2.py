# Machine Learning courseby Andrew Ng
# Python version
# Solution by: Amin Ahmadi
# Date: April 9, 2018

#### Logistic regression, classification
##################################################
import numpy as np
import matplotlib.pyplot as pl

# Read admission data into an numpy array for analysis and plotting
data = np.loadtxt('ex2data1.txt', delimiter=',', dtype=float)
pl.plot(data[:,0], data[:,1], '*b')

# To find admitted and rejected student

#admitted students
dp = data[data[:,2]==1,0:2]
#rejected students
dn = data[data[:,2]==0,0:2]


# Plot admitted and rejected student with different colors
# pl.plot(dp[:,0], dp[:,1], 'ko', label='admitted')
# pl.plot(dn[:,0], dn[:,1], 'b*', label='rejected')
# pl.legend()

########################################
def sigmoid(x):
    """Calculate sigmoid function of input paramter.
    input:
    ------
    x: float
    
    return:
    -------
    aux: float
    """
    if x == 0:
        aux=0.5
    else:
        aux = 1./( 1+np.exp(-x) )
        
    return aux
# to be applicable to an array, vectorize it 
sigmoid = np.vectorize(sigmoid)

########################################
def costFun(x,y,theta):
    """Calculate the cost function in logistic regression.
    input:
    ------
    x: array, float, features
    y: array, float, targets
    theta: array, float, hypothesis parameter
    
    return:
    -------
    float, the cost 
    """
    aux = 0
    M = x.shape[0]
    aux = np.sum(-y*np.log(sigmoid(np.dot(x,theta))) -\
                 (1-y)*np.log(1-sigmoid(np.dot(x,theta))))
    return aux/M

########################################
def costGrad(x,y,theta):
    """Gradient of the logistic regression cost function.
    input:
    ------
    x: array, float, features
    y: array, float, target
    theta: array, hypothesis paramters
    
    return:
    -------
    grad: array, float, dimension of features
    """
    M = x.shape[0]
    grad = np.zeros((x.shape[1]),float)
    temp = sigmoid(np.dot(theta,x.T))
    error = temp - y
    grad = (1./M)*np.dot(x.T,error)
    return grad

########################################
def g_des(x,y,theta_int,alpha):
    """Gradient desendent to find the minimum of the cost function.
    input:
    ------
    x: array, float, features
    y: array, float, target
    theta_int: array, hypothesis paramters, initial guess.
    
    return:
    -------
    theta:array, hypothesis parameters, fitted
    Jarr: array, cost function in each iteration
    """
    num_itr = 5000
    Jarr = np.zeros((num_itr), float)
    theta = theta_int
    print(Jarr.shape)
    for it in range(num_itr):
        theta -= alpha*costGrad(x,y,theta)
        Jarr[it] = costFun(x,y,theta)
    return theta, Jarr

########################################

x = data[:,0:2]
m = x.shape[0]
# add one column of 1s to the featurs for theta_0
x = np.hstack((np.ones((m,1), float),x))
y = data[:,2]
theta_int = np.array([-10, 0.1,0.1])
theta_f, J_arr = g_des(x,y,theta_int,0.001)
print(theta_f)
print(sigmoid(np.dot(theta_f,np.array([1,45,85],float))))
d1 = np.linspace(0,1,5000)
pl.plot(d1,J_arr, 'k,')


########################################
from scipy.optimize import minimize
# define one-paramter cost function for optimization
def cost_1param(theta):
    """one parameter cost function for optimization"""
    x = data[:,0:2]
    m = x.shape[0]
    x1 = np.ones((m,1),float)
    x = np.hstack((x1,x))
    y = data[:,2]
    return costFun(x,y,theta)

#optimization using "minimize" from scipy
res = minimize(cost_1param,(0.,0.1,0.1))
print(res)
print(res.x)
#prediction for test of (45,85) = 0.776
sigmoid(np.dot(res.x,np.array([1,45,85])))

########################################
x = data[:,0:2]
y = data[:,2]
theta = res.x
x_pos = x[y==1,:]
x_neg = x[y==0,:]
pl.plot(x_pos[:,0], x_pos[:,1], 'ko', label='admitted')
pl.plot(x_neg[:,0], x_neg[:,1], 'b*', label='rejected')
pl.legend()
x1_fit = np.linspace(min(x[:,0]), max(x[:,0]),100)
x2_fit = (-1./theta[2])* (theta[1]*x1_fit + theta[0])
pl.plot(x1_fit,x2_fit, 'r-')
pl.xlim(min(x[:,0]), max(x[:,0]))
pl.ylim(min(x[:,1]), max(x[:,1]))
pl.show()
