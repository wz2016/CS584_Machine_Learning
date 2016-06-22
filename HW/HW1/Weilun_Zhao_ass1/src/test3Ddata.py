import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

a1 =np.loadtxt('mvar-set1.dat')

a2 =np.loadtxt('mvar-set2.dat')

a3 =np.loadtxt('mvar-set3.dat')

a4 =np.loadtxt('mvar-set4.dat')

x1 = a1[:,[0]]
y1 = a1[:,[1]]
z1 = a1[:,[2]]
# print x1.size
# print x1
# x_train = x[:-20]
# x_test = x[-20:]

# y_train = y[:-20]
# y_test = y[-20:]

x2 = a2[:,[0]]
y2 = a2[:,[1]]
z2 = a2[:,[2]]

x3 = a3[:,[0]]
y3 = a3[:,[1]]
z3 = a3[:,[2]]

x4 = a4[:,[0]]
y4 = a4[:,[1]]
z4 = a4[:,[2]]

fig = plt.figure()

def data_subgraph(n, x, y,z, clr,title):
	# plt.subplot(2,2,n)
	# ax.plot(x, y, z, sp)
	ax = fig.add_subplot(n,projection='3d')
	ax.scatter(x,y,z,c=clr)
	# plt.title(title)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	# surf = ax.plot_surface(x, y, z)

def data_surfacec(n,x, y, z):
	ax = fig.add_subplot(n,projection='3d')
	x,y = np.meshgrid(x, y)
	surf = ax.plot_surface(x, y, z, color='b')

# data_subgraph(111, x1, y1, z1,'r', 'mvar-set1')
# data_surfacec(111, x1,y1, z1)
# data_subgraph(111, x2, y2, z2,'b', 'mvar-set2')
# data_subgraph(223, x3, y3, z3,'g', 'mvar-set3')
# data_subgraph(224, x4, y4, z4,'y', 'mvar-set4')

def graph_Surf(n, formula, x_range, y_range):
	ax = fig.add_subplot(n,projection='3d')
	x = np.array(x_range)
	y = np.array(y_range)
	z =eval(formula)
	x, y = np.meshgrid(x, y)
	surf = ax.plot_surface(x, y, z, color='b')

formula1 = '0.9959 + 0.9975*x + 0.9905*y'
formula2 = '0.0009+ 0.0646*x + -0.0006*y'
# graph_Surf(111, formula,range(-4, 4), range(-4, 4))
# data_subgraph(111, x1, y1, z1,'r', 'mvar-set1')
ax = fig.add_subplot(111,projection='3d')
# ax.scatter(x1,y1,z1,c='r')
ax.scatter(x2,y2,z2,c='r')
# plt.title(title)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x = np.array(range(-4, 4))
y = np.array(range(-4, 4))
z =eval(formula2)
x, y = np.meshgrid(x, y)
surf = ax.plot_surface(x, y, z, color='g')

plt.show()