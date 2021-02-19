import numpy as np

IMGPATH = "train-images-idx3-ubyte"
LBLPATH = "train-labels-idx1-ubyte"


TrainSamples = np.reshape(np.fromfile(IMGPATH, dtype=np.uint8, offset=16), (60000,28,28))

print(TrainSamples)