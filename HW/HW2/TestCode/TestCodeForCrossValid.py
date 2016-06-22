import numpy as np 
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print iris
print iris.data.shape, iris.target.shape