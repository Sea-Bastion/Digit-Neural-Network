#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:51:14 2023

@author: sebas
"""

import numpy as np
import json






def ReLU(input):
	return np.maximum(input, input*0.1)
	


def ReLUGradient(input):
    return np.where(input > 0, 1, 0.1)




#softmax function to limit output
def softmax(input):
	einput = np.exp(input)
	return einput/ einput.sum()





def softmaxGradient(SoftmaxOutput):
    output = -np.outer(SoftmaxOutput, SoftmaxOutput)
    np.fill_diagonal(output, SoftmaxOutput*(1-SoftmaxOutput))
    return output



#liner algibra for computing a single layer of NN
def RunLayer(input, weights, biases):
	return np.clip(weights.dot(input) + biases, -100, 100)






def cost(input, correct):
	return -(correct * np.log(input)).sum()


def costGradient(input, correct):
    return 2*(input - correct)



def SaveWeights(weights, biases, Arch, locations):
    exportData = {
        "Weights": [x.tolist() for x in weights],
        "Biases": [x.tolist() for x in biases],
        "Architecture": Arch.tolist()
    }
    
    JsonText = json.dumps(exportData, indent=4)
    
    with open(locations, 'w') as outfile:
        outfile.write(JsonText)
     





def LoadImages(ImgPath):
    with open(ImgPath, "rb") as images:

    	assert images.read(4) == b'\x00\x00\x08\x03'
    	ImgCount = int.from_bytes(images.read(4), "big")
    	ImgRows = int.from_bytes(images.read(4), "big")
    	ImgColumns = int.from_bytes(images.read(4), "big")
    	ImgSize = ImgRows * ImgColumns

    #load images into numpy 3d matrix
    return (np.reshape(np.fromfile(ImgPath, dtype=np.uint8, offset=16), (ImgCount, ImgSize)), ImgCount, ImgRows, ImgColumns)






def LoadLabels(LblPath):
    with open(LblPath, "rb") as labels:

    	assert labels.read(4) == b'\x00\x00\x08\x01'
    	LblCount = int.from_bytes(labels.read(4), "big")

    #load label data into numpy vector
    return (np.fromfile(LblPath, dtype=np.uint8, offset=8), LblCount)







def RunNetwork(inputs, Weights, Biases, Architecture, ActivationFuncs):
    
    #make space to store NN output
    #and init input layer with input
    RawLayerOutput = np.zeros( (Architecture.size), dtype=object )
    ActivatedOutput = np.zeros( (Architecture.size), dtype=object )
    RawLayerOutput[0] = inputs
    ActivatedOutput[0] = RawLayerOutput[0] #mapped to [0,1] so ReLU is redundent
    
    #run all layers
    for l in range(Architecture.size - 1):
        RawLayerOutput[l+1] = RunLayer(ActivatedOutput[l], Weights[l], Biases[l])
        ActivatedOutput[l+1] = ActivationFuncs[l]( RawLayerOutput[l+1] )
        
    return (ActivatedOutput, RawLayerOutput)