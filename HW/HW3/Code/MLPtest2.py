import  numpy as np 

class BackPropagationNetwork:

	layerCount = 0;
	shape = None
	weights = []

	def __init__(self, layerSize):
		#layer info
		self.layerCount = len(layerSize)-1;
		# self.layerCount = (layerSize)-1;
		self.shape = layerSize;

		#input/output data from last run
		self._layerInput=[]
		self._layerOutput = []
		self._previousWeightDelta = []

		#Create the weight arrays;
		for(l1, l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1, size = (l2,l1+1)))
			self._previousWeightDelta.append(np.zeros((l2,l1+1)));

	#run method
	def Run(self, input):
		lnCases = input.shape[0] #n put cases
		# clear out the previous intermediate value lists;
		self._layerInput=[];
		self._layerOutput = [];

		#run it;
		for index in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1,lnCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,lnCases])]))
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))
		return self._layerOutput[-1].T

	def TrainEpoch(self, input, target, trainingRate = 0.2,momentum=0.5):
		delta = []
		lnCases = input.shape[0]
		#first run the network;
		self.Run(input);

		for index in reversed(range(self.layerCount)):
			if index == self.layerCount -1:
				#compare to the target values;
				output_delta = self._layerOutput[index] - target.T
				error =np.sum(output_delta**2)
				delta.append(output_delta * self.sgm(self._layerInput[index], True))
			else:
				#compare to the following layer's delta
				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1, :]*self.sgm(self._layerInput[index],True))

		#compute weight deltas
		for index in range(self.layerCount):
			delta_index = self.layerCount-1 -index;

			if index == 0:
				layerOutput = np.vstack([input.T, np.ones([1,lnCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1,self._layerOutput[index-1].shape[1]])])

			curweightDelta = np.sum(

				) #?
			weightDelta = trainingRate * curweightDelta +momentum *self._previousWeightDelta[index]

			self.weights[index] -= weightDelta;
			self._previousWeightDelta = weightDelta
	#transfer function
	def sgm(self, x, Derivative=False):
		if not Derivative:
			return 1/(1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


if __name__=="__main__":
	bpn = BackPropagationNetwork((2,2,1))
	print bpn.shape
	print bpn.weights

	lvInput = np.array([[0,0],[1,1],[-1,0.5]])
	lvOutput = bpn.Run(lvInput)

	print "Input: {0}\n Output:{1} ".format(lvInput,lvOutput)

