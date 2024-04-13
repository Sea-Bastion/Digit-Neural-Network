
#http://yann.lecun.com/exdb/mnist/
#DFF type topology

import numpy as np
import Methods


#--------------------------------------------------------VARS---------------------------------------------------

IMGPATH = "Training Data/train-images-idx3-ubyte"
LBLPATH = "Training Data/train-labels-idx1-ubyte"

learnRate = 0.01
BatchSize = 1


#------------------------------------------------------main code------------------------------------------------------------

#---------------------load training data-------------------
#read sample image header data


#load images into numpy 3d matrix
TrainSamples, ImgCount, ImgRows, ImgColumns  = Methods.LoadImages(IMGPATH)
ImgSize = ImgColumns * ImgRows

#read simple label data for any problems
TrainLabels, LblCount = Methods.LoadLabels(LBLPATH)
assert LblCount == ImgCount


#----------------------Train NN--------------------

#make list of neurons in each layer
Architecture = np.array([ImgSize, 20, 20, 10])
LayerCount = Architecture.size
ConnectCount = LayerCount - 1

#init weights and biases 
# He initialization use normal dist with 2/[input count] sigma. good for ReLU
Weights = np.zeros( (ConnectCount), dtype=object )
Biases = np.zeros( (ConnectCount), dtype=object )
for l in range(ConnectCount):
	Weights[l] = np.random.normal(0, 2/Architecture[0], (Architecture[l+1], Architecture[l]) )
	Biases[l] = np.random.normal(0, 2/Architecture[0], (Architecture[l+1]) )

#ImgCount=20
#run though all batches
for y in range(ImgCount//BatchSize):
    
    #define deltas for batch
    WeightDelta1 = np.zeros(Weights[-1].shape)
    WeightDelta2 = np.zeros(Weights[-2].shape)
    WeightDelta3 = np.zeros(Weights[-3].shape)
    
    BiasDelta1 = np.zeros(Biases[-1].shape)
    BiasDelta2 = np.zeros(Biases[-2].shape)
    BiasDelta3 = np.zeros(Biases[-3].shape)
    
    
    #run batch
    for x in range(BatchSize):
        ImgID = y*BatchSize + x
    
        #get the target number for the NN
        Goal = np.zeros((10))
        Goal[TrainLabels[ImgID]] = 1
        
        InputData = TrainSamples[ImgID]/255
    
        
        #run the Neural Network
        ActivatedOutput, RawLayerOutput = Methods.RunNetwork(InputData, Weights, Biases, Architecture)

            
        #softmax final output
        outputData = ActivatedOutput[-1]
        
        
        #print cost
        print(Methods.cost(outputData, Goal))
        
        
        #calculate jacobian of softmax
        softmaxGrad = -np.outer(outputData, outputData)
        np.fill_diagonal(softmaxGrad, outputData*(1-outputData))
        
        #calcualte gradiant of ReLU
        ReLUGrad = np.asarray([ np.where(i < 0, 0.1, 1) for i in RawLayerOutput ], dtype=object )
        
        #find delta of output of each layer
        Del1 = 2*(outputData - Goal) * ReLUGrad[-1]
        Del2 = ( Del1 @ Weights[-1] ) * ReLUGrad[-2] 
        Del3 = ( Del2 @ Weights[-2] ) * ReLUGrad[-3]
        
        # watch for transposition when combining it's complex
        
        # find the delta for all weights and biases
        WeightDelta1 += -1*learnRate*np.outer(Del1, RawLayerOutput[-2])
        WeightDelta2 += -1*learnRate*np.outer(Del2, RawLayerOutput[-3])
        WeightDelta3 += -1*learnRate*np.outer(Del3, RawLayerOutput[-4])
        
        BiasDelta1 += -1*learnRate*Del1
        BiasDelta2 += -1*learnRate*Del2
        BiasDelta3 += -1*learnRate*Del3
        
        
    #apply deltas
    Weights[-3] += WeightDelta3
    Weights[-2] += WeightDelta2
    Weights[-1] += WeightDelta1
    
    Biases[-3] += BiasDelta3
    Biases[-2] += BiasDelta2
    Biases[-1] += BiasDelta1

print('done')


#ask to save generated data
save = input('Save Weights? (y/n): ')

#save data if yes
if 'y' in save.lower() :
    Methods.SaveWeights(Weights, Biases, Architecture, './latest.json')


# 10/8/23
# there seems to be an inverse relationship between cost and gradient max
# like if the cost is really big the gradient gets smaller. 
# I actually think this could be correct since the way the softmax functions works
# generally what you see is that one of the outputs is much larger then the others
# this causes a much higher error because it makes all the values near 0 except the high one which isn't the expected out
# however because its so high a nudge to that value or any value wouldn't actually change the output that much like a swaming effect
# this may be able to be fixed with better weight initialization

# that seemed to work I used HE initializations
# normal distrobution with a std of 2/[input count] supposed to work well with ReLU


# 10/10/23
# got the neural network to actually work however it was without softmax and without batches
# I think I can make it better. The two big problems are either it colapsing to only choosing one output
# or the cost exploding to infinity for an unknown reason. could be slingshotting.