from scipy.misc import imread

# img = imread("testImage.jpg")

# print img[0][0]
from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

mnist = fetch_mldata('MNIST original')
print mnist
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
X_train.shape
y_train.shape
size=len(y_train)

## extract "3" digits and show their average"
ind = [ k for k in range(size) if y_train[k]==3 ]
extracted_images=X_train[ind,:]

mean_image=extracted_images.mean(axis=0)
plt.imshow(mean_image.reshape(28,28))
plt.show()