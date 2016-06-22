import numpy as np
import myprime
import time

def primes(kmax):
	p = np.zeros(1000)
	result = []
	if kmax > 1000:
		kmax = 1000 
	k=0
	n=2
	while k < kmax:
		i=0
		while i < k and n % p[i] != 0:
			i=i+1 
		if i == k:
			p[k] = n 
			k=k+1 
			result.append(n)
		n=n+1 
	return result

start_time = time.time();
primes(1000000000);
print("--- Python function: %s seconds ---" % (time.time() - start_time));
start_time = time.time();
myprime.primes(1000000000);
print("--- Cython function: %s seconds ---" % (time.time() - start_time));
