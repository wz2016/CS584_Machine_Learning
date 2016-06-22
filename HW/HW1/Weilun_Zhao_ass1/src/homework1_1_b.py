import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

a1 =np.loadtxt('svar-set1.dat')
A1 = a1[:]

a2 =np.loadtxt('svar-set2.dat')
A2 = a2[:]

a3 =np.loadtxt('svar-set3.dat')
A3 = a3[:]

a4 =np.loadtxt('svar-set4.dat')
A4 = a4[:]

x = a1[:,[0]]
y = a1[:,[1]]
x_train = x[:-20]
x_test = x[-20:]
y_train = y[:-20]
y_test = y[-20:]

x2 = a2[:,[0]]
y2 = a2[:,[1]]
x2_train = x2[:-20]
x2_test = x2[-20:]
y2_train = y2[:-20]
y2_test = y2[-20:]

x3 = a3[:,[0]]
y3 = a3[:,[1]]
x3_train = x3[:-20]
x3_test = x3[-20:]
y3_train = y3[:-20]
y3_test = y3[-20:]

x4 = a4[:,[0]]
y4 = a4[:,[1]]
x4_train = x4[:-20]
x4_test = x4[-20:]
y4_train = y4[:-20]
y4_test = y4[-20:]


arr1 = np.array([0.26120329,1.98610257])
arr2 = np.array([0.33283915, -0.07891878])
arr3 = np.array([ 0.52697054, -0.0038674])
arr4 = np.array([0.96590264,-0.00315079])

def formula_form(arr):
	a0 = arr[0]
	a1 = arr[1]
	
	return '%f+%f*x_data'%(a0,a1)

def data_subgraph(n, x, y, sp,title,formula,x_range):
	plt.subplot(2,2,n)
	plt.plot(x, y, sp)
	plt.title(title)
	x_data=np.array(x_range)
	y_data = eval(formula)
	plt.plot(x_data, y_data)

data_subgraph(1, x_test, y_test, 'ro', 'svar-set1', formula_form(arr1), range(10, 30))
data_subgraph(2, x2_test, y2_test, 'bo', 'svar-set2',formula_form(arr2), range(0, 10) )
data_subgraph(3, x3_test, y3_test, 'go', 'svar-set3',formula_form(arr3), range(0, 10))
data_subgraph(4, x4_test, y4_test, 'yo', 'svar-set4',formula_form(arr4), range(0, 10))


plt.show()
