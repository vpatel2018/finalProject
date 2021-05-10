import random

#you can add additional fields if needed
#this is used to create a neural network node
class Neuron:
    def __init__(self, nextNodes=[], edgeWeights=[], error=0, output=0):
        self.nextNodes = nextNodes
        self.edgeWeights = edgeWeights
        self.error = error
        self.output = output
    #
#

class NeuralNetwork:
    
    #creates neural network
    def __init__(self, numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums):
        
        #NOTE: comments needed
    
        self.outputUnits = [Neuron() for x in range(numOutputUnits)]
        self.hiddenUnits = [Neuron(self.outputUnits, [random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numOutputUnits)]) for x in range(numHiddenUnits)]
        self.inputUnits = [Neuron(self.hiddenUnits, [random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numHiddenUnits)]) for x in range(numInputUnits)]

    #
    
    def updateOutputUnitErrors(self, targetValues):
        
        # \delta_k = (o_k) * (1 - o_k) * (t_k - o_k);

        for x in range(0, len(self.outputUnits)):
            self.outputUnits[x].error = (self.outputUnits[x].output) * (1 - self.outputUnits[x].output) * (targetValues[x] - self.outputUnits[x].output)
        #

    #
    
    def getSummForHiddenUnit(self, hiddenUnit):
        
        #\sum_{k \in outputs} (w_{kh} * \delta_k)
        
        total = 0
        edgeWeights = hiddenUnit.edgeWeights
        outputUnitErrors = [self.outputUnits[h].error for h in range(len(self.outputUnits))]
        
        for y in range(0, len(edgeWeights)):
            product = edgeWeights[y] * outputUnitErrors[y]
            total += product
        #
        
        return total
        
    #

    def updateHiddenUnitErrors(self):
        
        # \delta_h = (o_h) * (1 - o_h) * \sum_{k \in outputs} (w_{kh} * \delta_k)

        for x in range(0, len(self.hiddenUnits)):
            total = NeuralNetwork.getSummForHiddenUnit(self, self.hiddenUnits[x])
            output = self.hiddenUnits[x].output
            self.hiddenUnits[x].error = output * (1 - output) * total
        #   

    #
    
    def updateNetworkWeights(self, learningRate, inputVector):
        
        # w_{ji} = w_{ji} + \Delta w_{ji}
        # \Delta w_{ji} = \eta * \delta_j * x_{ji}
     
        pass
        
    #
    
    def doBackwardPropagation(self, targetValues, learningRate, inputVector):
        
        #may need more parameters
        
        NeuralNetwork.updateOutputUnitErrors(self, targetValues)
        NeuralNetwork.updateHiddenUnitErrors(self)
        NeuralNetwork.updateNetworkWeights(self, learningRate, inputVector)
           
    #
    
#
