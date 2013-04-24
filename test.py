from ocrn import dataset as ds
from ocrn import feature as ft
from ocrn import neuralnet as nn

import numpy
from pyfiglet import figlet_format as ascii_print

print "\n\n\n"
print "================================================================================"
print ascii_print('         Roaster OCRN', font='standard')
print "================================================================================"
print "\n"
print "Roaster Optical Character Recognition.\nFull-bodied, with an aroma of hazelnut. Version 0.1."
print "\n"

# Load up a neural network.
print "Butting our heads together...\n"
n = nn.neuralnet(100,80,1)

# Load up training set.
print "Pulling out our datasets...\n"
d = ds.dataset(100,1)

print "Generating...    ",
if d.generateDataSet():
	## Data set was successfully generated.
	print "Done!",
print "\n"

print "Loading...       ",
if n.loadTrainingData(d.getTrainingDataset()):
	## Data set successfully loaded.
	print "Done!",
print "\n"

while True:
	print "\n================================================================================\n"
	input = raw_input("Please select an option and hit <Return>: \n q: Quit \n t: Teach \n e: Test \n\n")
	if input == "q":
		## I'm outta here.
		break
	elif input == "t":
		# Ask how many times.
		t = int(raw_input("\nHow many times?\t:\t"))
		# Do it.
		n.teach(t)
	elif input == "e":
		# Ask for an input file.
		e = raw_input("\nEnter input file\t:\t")
		# Activate the NN with it! MAGIC!
		output = n.activate(ft.feature.getImageFeatureVector(e))
		print "\nThere is a high probability that the image is '" + str(unichr(output)) + "'\n"
	else:
		# Boo hoo.
		print "Invalid option\n"

print "\n================================================================================\n\n"
print ascii_print('Good-bye!', font='standard')
print "\n================================================================================"