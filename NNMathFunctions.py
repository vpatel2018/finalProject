import math
import random

def sigmoid(number):
  
    '''
    input: number represents a float
    output: result represents a float
    '''
    
    negatedNumber = number * -1
    denominator = math.pow(math.e, negatedNumber) + 1
    result = 1.0 / denominator
    
    return result
    
#

def initWeights(lowerLimit, upperLimit, sizeOfVector):
  
    #NOTE: comments needed
  
    weights = []
    
    for x in range(0, sizeOfVector):
        number = random.uniform(lowerLimit, upperLimit)
        weights.append(number) 
    #
    
    return weights
    
#

#NOTE: may have to test this function
def getErrorOfNN(trainingSet, outputUnitOutputs):
  
    #NOTE: comments needed
  
    error = 0
    
    for x in range(0, len(trainingSet)):
        targetValue = trainingSet[x][len(trainingSet[x]) - 1]
        errorForInstOfTrain = 0
        for y in outputUnitOutputs[x]:
            errorForInstOfTrain = errorForInstOfTrain + ((targetValue - y) * (targetValue - y))
        #
        error = error + errorForInstOfTrain
    #
    
    error = error * 0.5
    
    return error
    
#
