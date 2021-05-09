import math

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

def getErrorForOutputUnit():
  
    """
    \delta_k = (o_k) * (1 - o_k) * (t_k - o_k);
    """
    
    pass
  
#

def getErrorForHiddenUnit():
  
    """
    \delta_h = (o_h) * (1 - o_h) * \sum_{k \in outputs} (w_{kh} * \delta_k)
    """
    
    pass
  
#

def getWeightChange():
    
    """
    \Delta w_{ji} = \eta * \delta_j * x_{ji}
    \delta_g denotes the error term associated with unit g.
    \delta_g = - (\partial E_d / \partial net_g), where net_g = \sum_{i} w_{ji} * x_{ji}
    """
    
    pass
    
#
