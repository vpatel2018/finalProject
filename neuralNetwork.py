import random

#you can add additional fields if needed
#this is used to create a neural network node
class Neuron:
    def __init__(self, edgeWeights=[], inputs=[], error=0, output=0):
        self.edgeWeights = edgeWeights
        self.inputs = inputs
        self.error = error
        self.output = output
    #
#

class NeuralNetwork:
    
    #creates neural network
    def __init__(self, numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate):
        
        #NOTE: comments needed
    
        self.outputUnits = [Neuron() for x in range(numOutputUnits)]
        self.hiddenUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numOutputUnits)], [0 for y in range(numOutputUnits)]) for x in range(numHiddenUnits)]
        self.inputUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numHiddenUnits)], [0 for y in range(numHiddenUnits)]) for x in range(numInputUnits)]
        self.learningRate = learningRate

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
    
    def updateNetworkWeights(self):
        
        # w_{ji} = w_{ji} + \Delta w_{ji}
        # \Delta w_{ji} = \eta * \delta_j * x_{ji}
        
        for x in range(0, len(self.inputUnits)):
            for y in range(0, len(self.hiddenUnits)):
                delta = self.hiddenUnits[y].error
                inputValue = self.inputUnits[x].inputs[y]
                change = delta * inputValue * self.learningRate
                self.inputUnits[x].edgeWeights[y] = self.inputUnits[x].edgeWeights[y] + change
            #
        #
        
        for x in range(0, len(self.hiddenUnits)):
            for y in range(0, len(self.outputUnits)):
                delta = self.outputUnits[y].error
                inputValue = self.hiddenUnits[x].inputs[y]
                change = delta * inputValue * self.learningRate
                self.hiddenUnits[x].edgeWeights[y] = self.hiddenUnits[x].edgeWeights[y] + change
            #
        #
        
    #
    
    def doBackwardPropagation(self, targetValues):
        
        NeuralNetwork.updateOutputUnitErrors(self, targetValues)
        NeuralNetwork.updateHiddenUnitErrors(self)
        NeuralNetwork.updateNetworkWeights(self)
        
        for x in range(0, len(self.inputUnits)):
            self.inputUnits[x].inputs = [0] * len(self.inputUnits[x].inputs)
        #
        
        for x in range(0, len(self.hiddenUnits)):
            self.hiddenUnits[x].error = 0
            self.hiddenUnits[x].output = 0
            self.hiddenUnits[x].inputs = [0] * len(self.hiddenUnits[x].inputs)
        #
        
        for x in range(0, len(self.outputUnits)):
            self.outputUnits[x].error = 0
            self.outputUnits[x].output = 0
        #
           
    #
    
    def doForwardPropagation(self, inputVector):
        
        # Propagate input forward through the network.
        # compute the output o_u of every unit u (except input units I believe) in the entire network
        
        pass
        
    #
    
#
