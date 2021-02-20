
#http://yann.lecun.com/exdb/mnist/
#DFF type topology

import numpy as np

IMGPATH = "train-images-idx3-ubyte"
LBLPATH = "train-labels-idx1-ubyte"

HLAYERS = 2
HLAYERSIZE = 20
	

#----------------------------------------------------functions----------------------------------------------------------------

#container for ReLU function to add readability
def ReLU(value):
	return np.maximum(value, 0)
	
#softmax function to limit output
def softmax(input):
	return input/input.sum()

#liner algibra for computing a single layer of NN
def run_layer(inputs, weights, biases):
	return ReLU(weights.dot(inputs) + biases)


#------------------------------------------------------main code------------------------------------------------------------

#---------------------load training data-------------------
#read sample image header data
with open(IMGPATH, "rb") as images:

	assert images.read(4) == b'\x00\x00\x08\x03'
	ImgCount = int.from_bytes(images.read(4), "big")
	ImgRows = int.from_bytes(images.read(4), "big")
	ImgColumns = int.from_bytes(images.read(4), "big")
	ImgSize = ImgRows * ImgColumns

#load images into numpy 3d matrix
TrainSamples = np.reshape(np.fromfile(IMGPATH, dtype=np.uint8, offset=16), (ImgCount, ImgSize))

#read simple label data for any problems
with open(LBLPATH, "rb") as labels:

	assert labels.read(4) == b'\x00\x00\x08\x01'
	LblCount = int.from_bytes(labels.read(4), "big")
	assert LblCount == ImgCount

#load label data into numpy vector
TrainLabels = np.fromfile(LBLPATH, dtype=np.uint8, offset=8)


#----------------------run NN--------------------
Weights = np.random.rand(HLAYERSIZE, ImgSize)
Biases = np.random.rand(HLAYERSIZE)

out = softmax(run_layer(TrainSamples[0], Weights, Biases))
