import Image as im
import numpy as np
import os

class feature:
	#Return a 10X10 array from a monochrome BMP Image
	@staticmethod
	def getImageArray(imagepath):
		try:
			image = im.open(imagepath)
			imagearray=np.asarray(image.crop(image.getbbox()).resize((10,10))).astype(float)
			return imagearray
		except IOError:
			print "File Not Found"
			return np.zeros((10,10))
	
	# Normalizes a 2D array cell.
	@staticmethod
	def normalizeArrayCell( array, x, y):
		for c in range(1,5):
			for i in range(x-c,x+c+1):
				for j in range(y-c, y+c+1):
					if i<=9 and i>=0 and j<=9 and j>=0:
						if array[i,j] < array[x,y] and array[i,j] < array[x,y]-0.2*c:
							array[i,j]=array[x,y]-0.2*c
	
	# Returns a Normalized 2D Array from an Input Array. 
	# normalizeArrayCell is called for all non zero cells
	@staticmethod
	def getNormalizedArray( imagearray):
		nonzerocells = np.transpose(imagearray.nonzero())
		for i in range (0, nonzerocells.shape[0]):
			feature.normalizeArrayCell(imagearray, nonzerocells[i][0],nonzerocells[i][1])
		return imagearray
	
	# Returns Normalized 2D array from an Image
	@staticmethod
	def getNormalizedImageArray( imagepath):
		return feature.getNormalizedArray(feature.getImageArray(imagepath))
	
	# Returns Feature Vector of an Image
	@staticmethod
	def getImageFeatureVector( imagepath):
		array = feature.getNormalizedImageArray(imagepath)
		vector = np.resize(array,(1,100))
		return vector[0]


    @staticmethod
    def getTrainingCount():
        """
        Gets the number of trained images from the imageData file.
        """
        wcData = os.popen("wc -l ../data/imagedata").read()
        # Because wc returns number and filename
        return wcData.split()[0]
        

    @staticmethod
    def generateDataSetFromRoaster(dataTuple):
        """
        Takes a tuple of (imageData, asciiVal) and adds all images to
        ../data/trainingdata/ and then adds a line to imagedata
        """
        imageData, asciiVal = dataTuple
        
        # For each imageData entry in the imageData list, save as a bmp and
        # write that path to imageData with `path:asciiVal`

        #This is to not have to do a getTrainingCount call every time.
        trainCount = getTrainingCount()
        datafile = open("../data/imagedata", "a")

        for image in imageData:
            pathname = "../data/trainingdata/" + trainCount + ".bmp"
            image.save(pathname, "BMP")
            datafile.write(pathname+":"+asciiVal)
            trainCount++
        
       datafile.close()
