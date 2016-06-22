import numpy as np 
import math
a = 3;
b = 4;
c = 1;

d = max(a, b, c)
# print d 
# print "h" , "h"

a = np.array([1,2,3],dtype=np.float)
b = np.array([2,3,5], dtype=np.float)
c = a + b;
# print c/2

a = np.matrix([[1,2],[3,4]])
i = a.I;
d =  np.sqrt(a)

print np.linalg.det(d)
