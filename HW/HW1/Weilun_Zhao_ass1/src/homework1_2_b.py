import numpy as np
# x = np.array([[1,2,3],[2,3,4],[3,4,5]])
# x1 = x[:,[0]]
# print x1
# print x.shape
# y1 = np.array([[13,14,15]])
# y2 = np.array([[13],[14],[15],[16]])
# # print y.shape

# x = np.append(x, y1,axis = 0)
# # print x
# x = np.append(x, y2,axis = 1)
# print x
# print z2

# print x1 * y2
# a1 =np.loadtxt('mvar-set1.dat')
# A1 = a1[:]

# a2 =np.loadtxt('mvar-set2.dat')
# A2 = a2[:]

a3 =np.loadtxt('mvar-set3.dat')
A3 = a3[:]

a4 =np.loadtxt('mvar-set4.dat')
A4 = a4[:]


# def compute_error_for_line_given_points(a0, a1, a2, A):
# 	tot = A.size
# 	n = A[:, [0]].size
# 	m = tot/n
# 	xdata = A[:,0:m-1]
# 	ydata = A[:, [m-1]]
# 	totalError = 0

# 	for i in range(0, n):
# 		totalError+= (ydata[i]- (a0 + a1*xdata[i][0]+a2*xdata[i][1]))**2

# 	print "mse", totalError/n
# 	return totalError/n

def compute_error_for_line_given_points(a0, a1, a2, a3,a4, a5, A):
	tot = A.size
	n = A[:, [0]].size
	m = tot/n
	xdata = A[:,0:m-1]
	ydata = A[:, [m-1]]
	totalError = 0

	for i in range(0, n):
		totalError+= (ydata[i]- (a0 + a1*xdata[i][0]+a2*xdata[i][1]+a3*xdata[i][2]+a4*xdata[i][3]+a5*xdata[i][4]))**2

	print "mse", totalError/n
	return totalError/n

# def step_gradient(a0_current, a1_current, a2_current, A, learningRate):
# 	a0_gradient = 0
# 	a1_gradient = 0
# 	a2_gradient = 0

# 	tot = A.size
# 	n = A[:, [0]].size
# 	m = tot/n
# 	xdata = A[:,0:m-1]
# 	ydata = A[:, [m-1]]
# 	N = float(n)
# 	for i in range(0,n):
# 		x0 = xdata[i][0]
# 		x1 = xdata[i][1]
# 		y = ydata[i]
# 		# print x0, x1, y
# 		a0_gradient += -(2/N) * (y - (a0_current + (a1_current*x0)+(a2_current*x1)))
# 		a1_gradient += -(2/N) * x0*(y - (a0_current + (a1_current*x0)+(a2_current*x1)))
# 		a2_gradient += -(2/N) * x1*(y - (a0_current + (a1_current*x0)+(a2_current*x1)))
# 	# print a0_gradient, a1_gradient, a2_gradient
# 	new_a0 = a0_current - (learningRate*a0_gradient)
# 	new_a1 = a1_current - (learningRate*a1_gradient)
# 	new_a2 = a2_current - (learningRate*a2_gradient)
# 	# print new_a0,new_a1,new_a2
# 	return [new_a0, new_a1, new_a2]

def step_gradient(a0_current, a1_current, a2_current,a3_current,a4_current,a5_current, A, learningRate):
	a0_gradient = 0
	a1_gradient = 0
	a2_gradient = 0
	a3_gradient = 0
	a4_gradient = 0
	a5_gradient = 0

	tot = A.size
	n = A[:, [0]].size
	m = tot/n
	xdata = A[:,0:m-1]
	ydata = A[:, [m-1]]
	N = float(n)
	for i in range(0,n):
		x0 = xdata[i][0]
		x1 = xdata[i][1]
		x2 = xdata[i][2]
		x3 = xdata[i][3]
		x4 = xdata[i][4]
		y = ydata[i]
		# print x0, x1, y
		a0_gradient += -(2/N) * (y - (a0_current + (a1_current*x0)+(a2_current*x1)+ (a3_current*x2)+(a4_current*x3)+(a5_current*x4)))
		a1_gradient += -(2/N) * x0*(y - (a0_current + (a1_current*x0)+(a2_current*x1)+ (a3_current*x2)+(a4_current*x3)+(a5_current*x4)))
		a2_gradient += -(2/N) * x1*(y - (a0_current + (a1_current*x0)+(a2_current*x1)+ (a3_current*x2)+(a4_current*x3)+(a5_current*x4)))
		a3_gradient += -(2/N) * x2*(y - (a0_current + (a1_current*x0)+(a2_current*x1)+ (a3_current*x2)+(a4_current*x3)+(a5_current*x4)))
		a4_gradient += -(2/N) * x3*(y - (a0_current + (a1_current*x0)+(a2_current*x1)+ (a3_current*x2)+(a4_current*x3)+(a5_current*x4)))
		a5_gradient += -(2/N) * x4*(y - (a0_current + (a1_current*x0)+(a2_current*x1)+ (a3_current*x2)+(a4_current*x3)+(a5_current*x4)))
	# print a0_gradient, a1_gradient, a2_gradient
	new_a0 = a0_current - (learningRate*a0_gradient)
	new_a1 = a1_current - (learningRate*a1_gradient)
	new_a2 = a2_current - (learningRate*a2_gradient)
	new_a3 = a3_current - (learningRate*a3_gradient)
	new_a4 = a4_current - (learningRate*a4_gradient)
	new_a5 = a5_current - (learningRate*a5_gradient)

	# print new_a0,new_a1,new_a2
	return [new_a0, new_a1, new_a2,new_a3, new_a4, new_a5]
  
# def gradient_descent_runner(A, starting_a0, starting_a1,starting_a2, learning_rate, num_iterations):
# 	a0 = starting_a0
# 	a1 = starting_a1
# 	a2 = starting_a2
# 	for i in range(num_iterations):
# 		a0, a1, a2 = step_gradient(a0,a1, a2, A, learning_rate)
# 	return [a0, a1, a2]

def gradient_descent_runner(A, starting_a0, starting_a1,starting_a2, starting_a3, starting_a4,starting_a5,learning_rate, num_iterations):
	a0 = starting_a0
	a1 = starting_a1
	a2 = starting_a2
	a3 = starting_a3
	a4 = starting_a4
	a5 = starting_a5
	for i in range(num_iterations):
		a0, a1, a2, a3, a4, a5 = step_gradient(a0,a1, a2,a3,a4,a4, A, learning_rate)
	return [a0, a1, a2, a3, a4, a5]

def run():
	initial_a0 = 0
	initial_a1 = 0
	initial_a2 = 0
	initial_a3 = 0
	initial_a4 = 0
	initial_a5 = 0
	# compute_error_for_line_given_points(initial_a0, initial_a1, initial_a2,A1)
	compute_error_for_line_given_points(initial_a0, initial_a1, initial_a2,initial_a3, initial_a4,initial_a5, A3)

	# num_iterations = 1000
	num_iterations = 200
	# learning_rate = 0.0001
	learning_rate = 0.1
	# learning_rate = 0.000001
	print "num_iterations: ", num_iterations, " learning_rate: ",learning_rate
	# [new_a0, new_a1, new_a2] = gradient_descent_runner(A1, initial_a0, initial_a1,initial_a2, learning_rate, num_iterations)
	[new_a0, new_a1, new_a2, new_a3, new_a4,new_a5] = gradient_descent_runner(A3, initial_a0, initial_a1,initial_a2,initial_a3, initial_a4,initial_a5,  learning_rate, num_iterations)
	# print new_a0,new_a1,new_a2
	print new_a0, new_a1, new_a2, new_a3, new_a4,new_a5
	# compute_error_for_line_given_points(new_a0, new_a1, new_a2,A1)
	compute_error_for_line_given_points(new_a0, new_a1, new_a2, new_a3, new_a4,new_a5, A3)

if __name__ == '__main__':
    run()