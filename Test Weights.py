#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 21:43:04 2023

@author: sebas
"""

import numpy as np
import Methods
import json


WeightPath = "latest.json"

#---------------------------------------------------------Load Test Images---------------------------------

TestImages, ImgCount, ImgRows, ImgColumns = Methods.LoadImages("Training Data/t10k-images-idx3-ubyte")
TestLabels, LblCount = Methods.LoadLabels("Training Data/t10k-labels-idx1-ubyte")
assert LblCount == ImgCount
ImgSize = ImgRows * ImgColumns


#--------------------------------------------------------Load Weights and Biases---------------------------

with open(WeightPath, 'r') as infile:
    global RawData
    RawData = json.load(infile)
    
Weights = np.array([np.array(x) for x in RawData['Weights']], dtype=object)
Biases = np.array([np.array(x) for x in RawData['Biases']], dtype=object)
Architecture = np.array(RawData['Architecture'])

#---------------------------------------------------------Test Loop----------------------------------------

AmountRight = 0
CostSum = 0

for ImgID in range(ImgCount):
    
    #get input data from database
    InputData = TestImages[ImgID]/255
    
    #get goal vector from label
    Goal = np.zeros((10))
    Goal[TestLabels[ImgID]] = 1
    
    
    ActivatedOut, RawOut = Methods.RunNetwork(InputData, Weights, Biases, Architecture)
    
    OutputData = ActivatedOut[-1]
    
    
    CostSum += Methods.cost(OutputData, Goal)
    
    if np.argsort(OutputData)[-1] == TestLabels[ImgID]:
        AmountRight += 1
        
        
print( f'Percent Correct:  {100*AmountRight/ImgCount:.3f}%' )
print( f'Average Cost: {CostSum/ImgCount:.5f}' )
    
    