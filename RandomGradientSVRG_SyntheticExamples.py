import numpy as np
import matplotlib.pyplot as plt


n = 10**4
maxiterations = 10**2
d = 100
ell = 30
alpha = 1
xtrue = np.matrix(np.ones(d)).reshape([d,1])
memory = 20

## Generate A
A = np.random.randn(n,d)
Q, R = np.linalg.qr(A)
# decaying spectrum
spectrum = np.diag(np.ones(d)/np.linspace(1,d,d))

#non-decaying spectrum

#spectrum = np.diag(np.ones(d))
#
A = np.matrix(Q) * spectrum

## generate b
b = A*xtrue+.1*np.random.randn(n).reshape([n,1])

truth = np.linalg.pinv(A)*b

x = np.matrix(np.zeros((maxiterations,d)))
x[0,:] = np.random.uniform(-1,1,d)
error = np.zeros(maxiterations)

## Calculate Gradient
## Function = .5 *(Ax-b)**2; Gradient = A.T(A*x-b)
def grad(x):
    x=x.reshape([d,1])
    g=A.T*(A*x-b)
    return g

## Generate P
def getP():
    X = np.random.randn(d,ell)
    Q, R = np.linalg.qr(X)
    diagonal = np.diag(R)
    ph = diagonal/np.abs(diagonal)
    P = Q@(np.identity(ell)*ph)
    return(P)
#
# ei = np.array([[1],[0],[0]])
# tot = np.matrix(np.zeros([10,10]))
# for i in range(1,maxiterations):
#     P = getP()
#     tot = tot + P@ei@ei.T@P.T
#
# tot = tot/maxiterations

## SGD Version
for k in range(1,maxiterations):
    P = getP()
    x[k,:] = (x[k-1,:].reshape([d,1]) - alpha *P @ P.T @ grad(x[k-1,:])).T
    error[k] = np.log(np.linalg.norm(x[k,:].T-truth,2)/np.linalg.norm(truth,2))
plt.plot(error, label='Projected Gradient')

## SVRG Version with optimal stepsize parameter. Cheating!!
s = 0
for k in range(1,maxiterations):
    P = getP()
    newgrad = P @ P.T @ grad(x[k - 1, :])
    oldgrad = (P@P.T)@grad(x[s,:])

#    beta = float(1/(oldgrad.T*oldgrad)*oldgrad.T*(newgrad))
    beta = float((grad(x[k - 1, :]).T@grad(x[s,:]))/(grad(x[s,:]).T@grad(x[s,:])))
    x[k,:] = (x[k-1,:].reshape([d,1]) - alpha *(newgrad - beta*(P@P.T-np.diag(np.ones(d)))@grad(x[s,:])) ) .T
    if np.mod(k,memory)==0:
        s = k
    error[k] = np.log(np.linalg.norm(x[k,:].T-truth,2)/np.linalg.norm(truth,2))
plt.plot(error, label = 'Variance Reduced Proj. Gradient. Scalar eta')



## Luis Version for optimal stepsize.
## Cheating!!
s = 0
eta = np.zeros([d,1])
for k in range(1,maxiterations):
    P = getP()
    newgrad = P @ P.T @ grad(x[k - 1, :])
    oldgrad = (P@P.T)*grad(x[s,:])

#    beta = float(1/(oldgrad.T*oldgrad)*oldgrad.T*(newgrad))
    for j in range(0,d):
        eta[j] = ((P@P.T-np.diag(np.ones(d)))@grad(x[k-1,:]))[j]*((P@P.T-np.diag(np.ones(d)))@grad(x[s,:]))[j]/(((P@P.T-np.diag(np.ones(d)))@grad(x[s,:]))[j]**2)
    x[k,:] = (x[k-1,:].reshape([d,1]) - alpha *(newgrad - np.diagflat(eta)@(P@P.T-np.diag(np.ones(d)))@grad(x[s,:])) ) .T
    if np.mod(k,memory)==0:
        s = k
    error[k] = np.log(np.linalg.norm(x[k,:].T-truth,2)/np.linalg.norm(truth,2))
plt.plot(error, label = 'Variance Reduced Proj. Gradient. Diagonal matrix eta')

plt.legend()
plt.xlabel(r'$iteration$', fontsize = 14)
plt.ylabel(r'$\log \frac{\|\| x_k - x_*\|\|_2^2}{\|\| x_* \|\|_2^2}$', fontsize=14)

# ## SQN Version
# numerator = 0
# denominator = 0
# s = 0
# adding = 0
# for k in range(1, maxiterations):
#     P = getP()
#     newgrad = (P * P.T - np.diag(np.ones(d))) * grad(x[k - 1, :])
#     oldgrad = (P * P.T - np.diag(np.ones(d))) * grad(x[s, :])
#     adding  = 1/k * P*P.T* grad(x[k-1,:]) + (k-1)/k * adding
#     numerator = P*P.T * grad(x[k-1,:]) - adding
#     denominator = oldgrad
#     beta = numerator * oldgrad.T * np.linalg.pinv(oldgrad * oldgrad.T)
#     x[k, :] = (x[k - 1, :].reshape([d, 1]) - alpha * (P * P.T* grad(x[k - 1, :]) - beta * (oldgrad))).T
#     if np.mod(k, memory) == 0:
#         s = k
#     error[k] = np.log(np.linalg.norm(x[k, :].T - truth, 2) / np.linalg.norm(truth, 2))
#
# plt.plot(error)