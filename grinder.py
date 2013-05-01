from ocrn import dataset as ds
from ocrn import feature as ft
from ocrn import neuralnet as nn

from PIL import Image

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

import numpy
import pickle
import os
import socket
import md5
import datetime

hostname = socket.gethostname()
joey = hostname == "Plutonium"

OCRN_PATH = "/home/dg/Ocrn/"

if joey:
	OCRN_PATH = "/Users/josefdlange/Projects/Expresso/Ocrn/"

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
		

	def load_dataset(self, file_path=[OCRN_PATH+'data/inputdata']):
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
		self.neural_network.teach(maxEpochs)

	def train_loop(self, n):
		self.neural_network.teach(n)

	def guess(self, image_file_path):
		if image_file_path:
			feature_vector = ft.feature.getImageFeatureVector(image_file_path)
			result = self.neural_network.activate(feature_vector)
			print str(unichr(result))
			return str(unichr(result))

	def guess_on_image(self, image):
		if image:
			image = Image.open(image)
			pathname = OCRN_PATH+"/data/testdata/" + md5.new(str(datetime.datetime.now())).hexdigest() + ".bmp"
			tempImage = image.convert("1")
			tempImage.save(pathname, "BMP")
			result = self.guess(pathname)
			print result
			return result

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
		datafile = open(OCRN_PATH+"/data/inputdata", "a")

		for image in imageData:
			pathname = OCRN_PATH+"/data/trainingdata/" + str(trainCount) + ".bmp"
			tempImage = image.convert("1")
			tempImage.save(pathname, "BMP")
			datafile.write(pathname+":"+str(asciiVal)+"\n")
			trainCount = trainCount + 1 
		
		datafile.close()

	def getTrainingCount(self):
		"""
		Gets the number of trained images from the imageData file.
		"""
		# Will need to change this to relative path later.
		wcData = os.popen("wc -l " + OCRN_PATH + "/data/inputdata").read()
		# Because wc returns number and filename
		wcList = wcData.split()
		if not wcList:
			wcList = ['0']
		print "wcList is " + str(wcList)
		return int(wcList[0])

	def reset(self):
		self.neural_network = nn.neuralnet(100,80,1)
		self.data_set 		= ds.dataset(100,1)


def main():
	g = Grinder()
	print g
	print g.neural_network
	print g.neural_network.nnet
	print g.data_set
	print g.data_set.DS

if  __name__ =='__main__':
    main()
