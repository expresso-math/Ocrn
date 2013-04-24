from ocrn import dataset as ds
from ocrn import feature as ft
from ocrn import neuralnet as nn

import numpy
import pickle
import os

class Grinder:
	"""
	Class to encapsulate the text-recognition neural network, based off swvist's implementation.
	More details at http://github.com/expresso-math/Ocrn

	Important member variables:
	-- neural_network: 	The neural network. ocrn.neuralnet type.
	-- data_set:		The data set. ocrn.dataset type
	"""

	def __init__(self, clean=False):
		""" 
		Create a __new__ Grinder object.
		"""
		self.neural_network = nn.neuralnet(100,80,1)
		self.data_set 		= ds.dataset(100,1)
		if not clean:
			if os.path.isfile('neural_net_pickle.p'):
				file = open('neural_net_pickle.p', 'rb')
				self.neural_network = pickle.load(file)
				file.close()
			if os.path.isfile('data_set_pickle.p'):
				file = open('data_set_pickle.p', 'rb')
				self.data_set = pickle.load(file)
				file.close()

	def load_dataset(self, file_path=['/Users/josefdlange/Projects/Expresso/Ocrn/data/inputdata']):
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

	def train_loop(self, n):
		self.neural_network.teach(n)

	def guess(self, image_file_path):
		if image_file_path:
			feature_vector = ft.feature.getImageFeatureVector(image_file_path)
			result = self.neural_network.activate(feature_vector)
			return str(unichr(result))

	def guess_on_image(self, image):
		if image:
			vector = ft.feature.getImageFeatureVectorForLoadedFile(image)
			result = self.neural_network.activate(vector)
			print result
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

	def generateDataSetFromRoaster(self, dataTuple):
		"""
		Takes a tuple of (imageData, asciiVal) and adds all images to
		../data/trainingdata/ and then adds a line to imagedata
		"""
		imageData, asciiVal = dataTuple
		
		# For each imageData entry in the imageData list, save as a bmp and
		# write that path to imageData with `path:asciiVal`

		#This is to not have to do a getTrainingCount call every time.
		trainCount = self.getTrainingCount()
		datafile = open("/Users/josefdlange/Projects/Expresso/Ocrn/data/inputdata", "a")

		for image in imageData:
			pathname = "/Users/josefdlange/Projects/Expresso/Ocrn/data/trainingdata/" + str(trainCount) + ".bmp"
			tempImage = image.convert("L")
			tempImage.save(pathname, "BMP")
			datafile.write(pathname+":"+str(asciiVal)+"\n")
			trainCount = trainCount + 1 
		
		datafile.close()

	def getTrainingCount(self):
		"""
		Gets the number of trained images from the imageData file.
		"""
		# Will need to change this to relative path later.
		wcData = os.popen("wc -l /Users/josefdlange/Projects/Expresso/Ocrn/data/inputdata").read()
		# Because wc returns number and filename
		wcList = wcData.split()
		print wcList
		return int(wcList[0])


def main():
	g = Grinder()
	g.load_dataset()
	# g.train_loop(10000)
	g.train_to_convergence()
	# epochs = 0
	# value = 1
	# while value > 0.001:
	# 	value = g.neural_network.teach_one()
	# 	print str(epochs) + ' : ' + str(value)
	# 	epochs = epochs + 1
	# print 'Took ' + str(epochs) + ' epochs...'
	print g.guess('/Users/josefdlange/Projects/Expresso/Ocrn/data/testdata/t1.bmp')
	print g.guess('/Users/josefdlange/Projects/Expresso/Ocrn/data/testdata/t2.bmp')
	print g.guess('/Users/josefdlange/Projects/Expresso/Ocrn/data/testdata/t3.bmp')
	print g.guess('/Users/josefdlange/Projects/Expresso/Ocrn/data/testdata/t4.bmp')
	print g.guess('/Users/josefdlange/Projects/Expresso/Ocrn/data/testdata/t5.bmp')
	print g.guess('/Users/josefdlange/Projects/Expresso/Ocrn/data/testdata/t6.bmp')

if  __name__ =='__main__':
    main()
