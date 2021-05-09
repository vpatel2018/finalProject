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

#NOTE: may have to test this function
def getErrorOfNN():
  
    #NOTE: comments needed
    #parameters needed
  
    pass
    
#

def getErrorOfOutputUnit(outputValue, targetValue):
    
    return (outputValue) * (1 - outputValue) * (targetValue - outputValue)
    
#

def getErrorOfHiddenUnit(outputValue, connectionWeights, outputUnitErrors):
 
    total = 0
  
    for x in range(0, len(connectionWeights)):
        product = connectionWeights[x] * outputUnitErrors[x]
        total = total + product
    #
    
    return (outputValue) * (1 - outputValue) * total
  
#
