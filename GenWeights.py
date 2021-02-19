#http://yann.lecun.com/exdb/mnist/

import numpy as np

IMGPATH = "train-images-idx3-ubyte"
LBLPATH = "train-labels-idx1-ubyte"
	
#read sample image header data
with open(IMGPATH, "rb") as images:

	assert images.read(4) == b'\x00\x00\x08\x03'
	ImgCount = int.from_bytes(images.read(4), "big")
	ImgRows = int.from_bytes(images.read(4), "big")
	ImgColumbs = int.from_bytes(images.read(4), "big")
	ImgSize = ImgRows * ImgColumbs

#load images into numpy 3d matrix
TrainSamples = np.reshape(np.fromfile(IMGPATH, dtype=np.uint8, offset=16), (ImgCount, ImgRows, ImgColumbs))

#read simple label data for any problems
with open(LBLPATH, "rb") as labels:
	
	assert labels.read(4) == b'\x00\x00\x08\x01'
	LblCount = int.from_bytes(labels.read(4), "big")
	assert LblCount == ImgCount

#load label data into numpy vector
TrainLabels = np.fromfile(LBLPATH, dtype=np.uint8, offset=8)

