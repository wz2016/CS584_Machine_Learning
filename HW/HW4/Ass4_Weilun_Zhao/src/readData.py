import numpy as np 

# dataFile = open("dataFileNonLS.txt", "r");
# dataFile = open("dataFile.txt", "r");
# dataFile = open("dataFile2.txt", "r");
dataFile = open("dataFileKernel.txt", "r");
dataFile.seek(0)
c = np.load(dataFile)

print c