from ocrn import dataset as ds
from ocrn import feature as ft
from ocrn import neuralnet as nn

import numpy
import pickle
from pyfiglet import figlet_format as ascii_print

class Grinder:
	"""
	Class to encapsulate the text-recognition neural network, based off swvist's implementation.
	More details at http://github.com/expresso-math/Ocrn

	Important member variables:
	-- neural_network: 	The neural network. ocrn.neuralnet type.
	-- data_set:		The data set. ocrn.dataset type
	"""

	def __init__(self):
		""" 
		Create a __new__ Grinder object.
		"""
		self.neural_network = nn.neuralnet(100,80,1)
		self.data_set 		= ds.dataset(100,1)

	def load_dataset(self, file_path=['data/inputdata']):
		"""
		Load the dataset from file. Defaults to the above.
		"""
		if self.data_set.generateDataSetFromFile(file_path):
			print "Successfully loaded dataset from file."
			if self.neural_network.loadTrainingData(self.data_set.getTrainingDataset()):
				print "Successfully loaded Training Data from data_set."
			else:
				# Couldn't load training data from data set.
				print "There was an error loading the training data into the neural network."
		else:
			# Couldn't load data set.
			print "Something is really broken, since this method always returns 1."

	def train_to_convergence(self):
		self.neural_network.teachUntilConvergence()

	def train(self, maxEpochs = 100):
		self.neural_network.teachUntilConvergence(max=maxEpochs)

	def guess(self, image_file_path):
		if image_file_path:
			feature_vector = ft.feature.getImageFeatureVector(image_file_path)
			result = self.neural_network.activate(feature_vector)
			return str(unichr(result))

	def pickle_network(self):
		"""
		Pickle the neural net.
		"""
		file = open('neural_net_pickle.p', 'wb')
		pickle.dump(self.neural_network, file)
		file = open('data_set_pickle.p', 'wb')
		pickle.dump(self.data_set, file)

	def unpickle_network(self):
		"""
		Unpack the neural net from a pickle.
		"""
		file = open('neural_net_pickle.p', 'rb')
		self.neural_network = pickle.load(file)
		file = open('data_set_pickle.p', 'rb')
		self.data_set = pickle.load(file)