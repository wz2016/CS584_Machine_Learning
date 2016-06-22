import numpy as np

feature = np.array([[1,2,3,4],[1,2,3,4],[2,3,4,5],[2,4,5],[1,2,3],[1,2,3]])

print feature[0][0]


# d = {feature[0][0]:12, feature[0][1]:2}
# print d[1]
d = {}
d[1] = 2
print d.values()
n = np.zeros(10);
# n[0] = d
print n
d[4] = 12
for i in range(0,2):
	for j in range(0,3):
		d[str(i)+ str(j)] = i+j

c = str(1)+str(2)
print c[1]
print d
print d['00']