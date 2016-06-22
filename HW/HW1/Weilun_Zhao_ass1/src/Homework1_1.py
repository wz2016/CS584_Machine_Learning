import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.interpolate import *

a1 =np.loadtxt('svar-set4.dat')

x = a1[:,[0]]
y = a1[:,[1]]

x_train = x[:-50]
x_test = x[-50:]

y_train = y[:-50]
y_test = y[-50:]

# print x_test
plt.plot(x,y, 'ro')
# plt.show()

n = x_train.size
m_test = x_test.size
print "x_train.size: ", n
print "x_test.size: ", m_test
a = 0
#sum of x
for i in range(0,n):
	a += x[i]
# print a
#sum of y
b = 0
for i in range(0, n):
	b+=y[i]
# print b
#sum of x*y
c = 0
for i in range(0, n):
	c+=x[i]*y[i]
# print c
#sum of x^2
d = 0
for i in range(0, n):
	d+=x[i]*x[i]
# print d

# print n, a, d, b,c
q = np.array([[n, a],[a,d]])
p = np.array([b,c])
w = np.linalg.solve(q,p)

# print w

k1 = w[0]; k2 = w[1]

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y, linewidth = 1.5)  
    # plt.show()

graph('k1 + k2*x', range(0, 11))
# print k1, k2
plt.xlabel("x")
plt.ylabel("y")

p = x_train.reshape((n))
q = y_train.reshape((n))

s = 0
for i in range(0,n):
	s += (k1 + k2*x_train[i] - y_train[i])**2
	# print i

# s = s/(n)
# print s
SStotal= len(y_train)*np.var(y_train)

# print (SStotal)
print "p1 my method: "
rsq= 1 - s/SStotal
print"rsq_train: ",rsq

s = 0
n_test = x_test.size
for i in range(0,n_test):
	s += (k1 + k2*x_test[i] - y_test[i])**2
	# print i

# s = s/(n_test)
# print s
SStotal= len(y_test)*np.var(y_test)

rsq= 1 - s/SStotal
print"rsq_train: ",rsq

# plt.show()

# p = x_train.reshape((n))
# q = y_train.reshape((n))


p1 = np.polyfit(p, q, 1)
p2 = np.polyfit(p, q, 2)
p3 = np.polyfit(p, q, 3)
p4 = np.polyfit(p, q, 4)
p5 = np.polyfit(p, q, 5)

print "******************************"
print "p1: ",p1
yfit = p1[0]*x_train+p1[1]
# print(yfit)

yresid = y_train - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_train: ", SSresid/n
SStotal= len(y_train)*np.var(y_train)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_train: ",rsq


#
yfit = p1[0]*x_test+p1[1]
# print(yfit)

yresid = y_test - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_test: ", SSresid/m_test
SStotal= len(y_test)*np.var(y_test)

print (SSresid/SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_test: ",rsq
graph('p1[0]*x+p1[1]', range(0,11))

print "******************************"
print "p2: ",p2
yfit = p2[0]*(x_train)**2+p2[1]*x_train+p2[2]
# print(yfit)

yresid = y_train - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_train: ", SSresid/n
SStotal= len(y_train)*np.var(y_train)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_train: ",rsq
graph('p2[0]*x**2+p2[1]*x+p2[2]',range(0,11))

#
yfit = p2[0]*(x_test)**2+p2[1]*x_test+p2[2]
# print(yfit)

yresid = y_test - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_test: ", SSresid/m_test
SStotal= len(y_test)*np.var(y_test)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_test: ",rsq


print "******************************"
print "p3: ",p3
yfit = p3[0]*(x_train)**3+p3[1]*(x_train**2)+p3[2]*x_train+p3[3]
# print(yfit)

yresid = y_train - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_train: ", SSresid/n
SStotal= len(y_train)*np.var(y_train)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_train: ",rsq

#
yfit = p3[0]*(x_test)**3+p3[1]*(x_test**2)+p3[2]*x_test+p3[3]# print(yfit)

yresid = y_test - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_test: ", SSresid/m_test
SStotal= len(y_test)*np.var(y_test)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_test: ",rsq
graph('p3[0]*(x)**3+p3[1]*(x**2)+p3[2]*x+p3[3]',range(0,11))


print "******************************"
print "p4: ",p4
yfit = p4[0]*(x_train)**4+p4[1]*(x_train)**3+p4[2]*(x_train**2)+p4[3]*x_train+p4[4]
# print(yfit)

yresid = y_train - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_train: ", SSresid/n
SStotal= len(y_train)*np.var(y_train)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_train: ",rsq


#
yfit = p4[0]*(x_test)**4+p4[1]*(x_test)**3+p4[2]*(x_test**2)+p4[3]*x_test+p4[4]

yresid = y_test - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_test: ", SSresid/m_test
SStotal= len(y_test)*np.var(y_test)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_test: ",rsq
graph('p4[0]*(x)**4+p4[1]*(x)**3+p4[2]*(x**2)+p4[3]*x+p4[4]',range(0,11))

print "******************************"
print "p5: ",p5
yfit = p5[0]*(x_train)**5+p5[1]*(x_train)**4+p5[2]*(x_train)**3+p5[3]*(x_train**2)+p5[4]*x_train+p5[5]
# # print(yfit)

yresid = y_train - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_train: ", SSresid/n
SStotal= len(y_train)*np.var(y_train)

# print (SStotal)
# rsq= 1 - SSresid/SStotal
print"rsq_train: ",rsq


# #
yfit = p5[0]*(x_test)**5+p5[1]*(x_test)**4+p5[2]*(x_test)**3+p5[3]*(x_test**2)+p5[4]*x_test+p5[5]
yresid = y_test - yfit

SSresid = sum(pow(yresid, 2))
print "MSE_test: ", SSresid/m_test
SStotal= len(y_test)*np.var(y_test)

# print (SStotal)
rsq= 1 - SSresid/SStotal
print"rsq_test: ",rsq
# graph('p5[0]*(x)**5+p5[1]*(x)**4+p5[2]*(x)**3+p5[3]*(x**2)+p5[4]*x+p5[5]',range(0,11))
# plt.legend()
plt.show()
