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

def getErrorForOutputUnit(o_k, t_k):
  
    #NOTE: comments needed
  
    """
    \delta_k = (o_k) * (1 - o_k) * (t_k - o_k)
    """
    
    return (o_k) * (1.0 - o_k) * (t_k - o_k)
  
#

def getErrorForHiddenUnit(o_h, weights, deltaKs):
  
    #NOTE: comments needed
      
    """
    \delta_h = (o_h) * (1 - o_h) * \sum_{k \in outputs} (w_{kh} * \delta_k)
    """
    
    total = 0
    
    for r in range(0, len(weights)):
        product = weights[r] * deltaKs[r]
        total = total + product
    #
    
    result = (o_h) * (1 - o_h) * (total)
    
    return result
  
#

def getWeightChange():
  
    #NOTE: this function needs parameters
    
    """
    \Delta w_{ji} = \eta * \delta_j * x_{ji}
    \delta_g denotes the error term associated with unit g.
    \delta_g = - (\partial E_d / \partial net_g), where net_g = \sum_{i} w_{ji} * x_{ji}
    """
    
    pass
    
#
