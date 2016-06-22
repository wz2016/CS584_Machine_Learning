from cvxopt import matrix, normal
from cvxopt import solvers
# from cvxopt.lapack import gesv, getrs
from cvxopt.lapack import gbsv, gbtrf, gbtrs
import numpy as np 
# n=10
# A=normal(n,n)
# b = normal(n)
# ipiv=matrix(0,(n,1))
# # print ipiv
# x=+b
# gesv(A,x,ipiv)

# x2=+b
# getrs(A,ipiv,x2,trans='T')
# x+=x2
# print b

# n, kl = 4, 2
# A=matrix([[0., 1., 3., 6.], [2., 4., 7., 10.], [5., 8., 11., 0.], [9., 12., 0., 0.]])
# x = matrix(1.0, (n,1))
# print x
# gbsv(A, kl, x)
# print x

# Q = 2*matrix([ [2, .5], [.5, 1] ])
# p = matrix([1.0, 1.0])
# G = matrix([[-1.0,0.0],[0.0,-1.0]])
# h = matrix([0.0,0.0])
# A = matrix([1.0, 1.0], (1,2))
# b = matrix(1.0)
# print b
# sol=solvers.qp(Q, p, G, h, A, b)

# print sol

import matplotlib.pyplot as plt
import numpy as np
rad = 1
num = 100

t = np.random.uniform(0.0, 2.0*np.pi, num)
r = rad * np.sqrt(np.random.uniform(0.0, 1.0, num))
x = r * np.cos(t)
y = r * np.sin(t)

plt.plot(x, y, "ro", ms=1)
plt.axis([-2, 2, -2, 2])
plt.show()

s = np.random.uniform(-1,0,1000)
print s
