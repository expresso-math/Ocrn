from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

class neuralnet:
	def __init__(self, *args):
		if len(args) >=2 :
			self.nnet = buildNetwork(*args)
			self.inputdimension = args[0]
			self.outputdimension = args[len(args)-1]
			self.nnet.sortModules()
			print self.inputdimension
			print self.outputdimension
		else:
			print "Number of layers must be greater than or equal to two\n"

	# Loads the training data into neural network before training.
	def loadTrainingData(self, trainingdataset):
		if trainingdataset.getDimension('input') == self.inputdimension and \
			trainingdataset.getDimension('target') == self.outputdimension:
				self.trainer = BackpropTrainer(self.nnet, trainingdataset, learningrate = 0.0001, verbose = True) ##, learningrate = 0.001, lrdecay = 1.0, momentum = 0.0, verbose=True, batchlearning=False, weightdecay=0.001)
				return 1
		else:
			print "Dataset-Network size mismatch\n"
			return 0

	# Train the neural network 'n' times with loaded training data.
	def teach(self, n):
		self.teachUntilConvergence(max=n)
		# for i in range (1, n+1):
		# 	str(self.trainer.train())


	def teach_one(self):
		return self.trainer.train()

	# Train until convergence.
	def teachUntilConvergence(self, max=0):
		if max > 0:
			self.trainer.trainUntilConvergence(maxEpochs=100, verbose=True, validationProportion=0.3)
		else:
			self.trainer.trainUntilConvergence(verbose=True)

	# Activate the Neural Network with test data. Returns calculated output.
	def activate(self, testdata):
		if testdata.size == self.inputdimension:
			x = self.nnet.activate(testdata)
			return int(round(x[0],0))
		else:
			print "Test data error\n"
			return 0
