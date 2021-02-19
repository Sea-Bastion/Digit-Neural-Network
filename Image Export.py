import numpy as np
from PIL import Image
import sys

PATH = sys.argv[1]
INDEX = int(sys.argv[2])


with open(PATH, "rb") as images:
	
	#get header data
	assert images.read(4) == b'\x00\x00\x08\x03'
	ImgCount = int.from_bytes(images.read(4), "big")
	ImgRows = int.from_bytes(images.read(4), "big")
	ImgColumbs = int.from_bytes(images.read(4), "big")
	ImgSize = ImgRows * ImgColumbs
	
	images.seek(16 + ImgSize * INDEX)
	
	ImageData = np.zeros((ImgColumbs, ImgRows, 3), dtype=np.uint8)
	for x in range(0, ImgColumbs):
		for y in range(0, ImgRows):
			pixel = 255 - int.from_bytes(images.read(1), "big")
			ImageData[x,y] = [pixel,pixel,pixel]
			
	

Image.fromarray(ImageData).save("image" + str(INDEX) +".png" )
	
	