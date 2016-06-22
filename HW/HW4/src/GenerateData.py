import numpy as np 

def generateKernelData():
	# n = 50; #data size (50*2, 2)
	# a = np.random.random([2])
	# for i in range(1, n):
	# 	a = np.row_stack((a,np.random.random([2])))

	# a = np.column_stack((a, np.zeros(n).reshape(n,1)-1)) 

	# # print a
	# b = -np.random.random([2])
	# for i in range(1, n):
	# 	b = np.row_stack((b,-np.random.random([2])))

	# b = np.column_stack((b, np.zeros(n).reshape(n,1)+1))

	# c = np.row_stack((a,b))
	rad = 2
	num = 100

	t = np.random.uniform(0.0, 2.0*np.pi, num)
	r = rad * np.sqrt(np.random.uniform(0.0, 1.0, num))
	x = r * np.cos(t)
	y = r * np.sin(t)
	c = np.column_stack((x,y))

	label = np.zeros((100))
	# print label.shape
	for i in range(num):
		if ((x[i])**2+(y[i])**2) > 0.2:
			label[i] = 1;
		else:
			label[i] = -1;

	c = np.column_stack((c, label))
	print c
	dataFile = open("dataFileKernel_SubExm.txt", "w+");
	np.save(dataFile,c);

	dataFile.seek(0)
	print np.load(dataFile)
	print "create data matrix:", c.shape

def main():
	generateKernelData()

if __name__=="__main__":
	main()