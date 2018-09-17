## SVRG find optimal stepsize for a given m, gamma, lambda
import numpy as np
from matplotlib import pyplot as plt
m = 700
gamma = .01
lamb = .1

def getf(alpha):
    f = 1 / (m * gamma * alpha * (1 - 2 * lamb * alpha)) + 2 * lamb * alpha / (1 - 2 * lamb * alpha)
    return(f)

def getGrad(alpha):
    grad = (2 * alpha**2 * gamma * lamb * m + 4 * alpha * lamb - 1)/ (alpha**2 * gamma * m * (1 - 2 * alpha * lamb)**2)
    return(grad)

alpha = .03
for i in np.arange(1,100000):
    alpha = alpha - .001*getGrad(alpha)

if (getf(alpha)> 1):
    print('Error, increase m')
else:
    print(getf(alpha))


## Random Gradient SVRG
## find optimal step sizes and optimal sample rate for Random Gradient SVRG
import numpy as np
from matplotlib import pyplot as plt
m = 1200
gamma = .01
lamb = 1
for rho in np.linspace(.5,1.1,100):


    def getf(alpha):
        f = 1 / (m * gamma * alpha * (1 - 2 * lamb * alpha* rho**2)) + 2 * lamb * alpha*(rho**2+1-2*rho) / (1 - 2 * lamb * alpha*rho**2)
        return(f)

    def getGrad(alpha):
        grad = (2 * alpha**2 * gamma * lamb * m *(rho-1)**2+ 4 * alpha * lamb*rho**2 - 1)/ (alpha**2 * gamma * m * (1 - 2 * alpha * lamb*rho**2)**2)
        return(grad)

    alpha = .03
    for i in np.arange(1,10000):
        alpha = alpha - .001*getGrad(alpha)

    if (getf(alpha)> 1):
        print('Error, increase m')
    else:
        print(getf(alpha)*1/rho)
    plt.scatter(rho, getf(alpha)/rho)