import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.interpolate import *


a1 =np.loadtxt('svar-set1.dat')

a2 =np.loadtxt('svar-set2.dat')

a3 =np.loadtxt('svar-set3.dat')

a4 =np.loadtxt('svar-set4.dat')

x = a1[:,[0]]
y = a1[:,[1]]

x_train = x[:-20]
x_test = x[-20:]

y_train = y[:-20]
y_test = y[-20:]

x2 = a2[:,[0]]
y2 = a2[:,[1]]

x3 = a3[:,[0]]
y3 = a3[:,[1]]

x4 = a4[:,[0]]
y4 = a4[:,[1]]

def data_subgraph(n, x, y, sp,title):
	plt.subplot(2,2,n)
	plt.plot(x, y, sp)
	plt.title(title)

data_subgraph(1, x, y, 'ro', 'svar-set1')
data_subgraph(2, x2, y2, 'bo', 'svar-set2')
data_subgraph(3, x3, y3, 'go', 'svar-set3')
data_subgraph(4, x4, y4, 'yo', 'svar-set4')



plt.show()
