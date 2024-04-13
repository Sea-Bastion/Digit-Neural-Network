#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:51:14 2023

@author: sebas
"""

import numpy as np
import json


def ReLU(value):
	return np.maximum(value, value*0.1)
	
#softmax function to limit output
def softmax(input):
	einput = np.exp(input - np.amax(input))
	return np.divide(einput, einput.sum())

#liner algibra for computing a single layer of NN
def RunLayer(inputs, weights, biases):
	return weights.dot(inputs) + biases

def cost(inputs, correct):
	return np.square(inputs - correct).sum()

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


def RunNetwork(inputs, Weights, Biases, Architecture):
    
    #make space to store NN output
    #and init input layer with input
    RawLayerOutput = np.zeros( (Architecture.size), dtype=object )
    ActivatedOutput = np.zeros( (Architecture.size), dtype=object )
    RawLayerOutput[0] = inputs
    ActivatedOutput[0] = RawLayerOutput[0] #mapped to [0,1] so ReLU is redundent
    
    #run all layers
    for l in range(Architecture.size - 1):
        RawLayerOutput[l+1] = RunLayer(ActivatedOutput[l], Weights[l], Biases[l])
        ActivatedOutput[l+1] = ReLU( RawLayerOutput[l+1] )
        
    return (ActivatedOutput, RawLayerOutput)