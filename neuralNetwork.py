import random
import math

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
           
    #
    
    def updateOutputForHiddenUnit(self, index):
        
        total = 0
        
        for x in range(0, len(self.inputUnits)):
            inputValue = self.inputUnits[x].inputs[index]
            weight = self.inputUnits[x].edgeWeights[index]
            product = inputValue * weight
            total += product
        #
        
        self.hiddenUnits[index].output = total
           
    #
    
    def updateOutputForOutputUnit(self, index):
        
        total = 0
        
        for x in range(0, len(self.hiddenUnits)):
            inputValue = self.hiddenUnits[x].inputs[index]
            weight = self.hiddenUnits[x].edgeWeights[index]
            product = inputValue * weight
            total += product
        #
        
        self.outputUnits[index].output = total
        
    #
    
    def sigmoid(self, output):
        
        negatedOutput = -1 * output
        denominator = 1 + math.pow(math.e, negatedOutput)
        result = (1.0) / (1.0 * denominator)
            
        return result
               
    #
    
    def doForwardPropagation(self, inputVector):
        
        for x in range(0, len(self.inputUnits)):
            self.inputUnits[x].inputs = [inputVector[x]] * len(self.inputUnits[x].inputs) 
        #
        
        for x in range(0, len(self.hiddenUnits)):
            NeuralNetwork.updateOutputForHiddenUnit(self, x)
            self.hiddenUnits[x].output = NeuralNetwork.sigmoid(self, self.hiddenUnits[x].output)
            self.hiddenUnits[x].inputs = [self.hiddenUnits[x].output] * len(self.hiddenUnits[x].inputs)
        #
        
        for x in range(0, len(self.outputUnits)):
            NeuralNetwork.updateOutputForOutputUnit(self, x)
            self.outputUnits[x].output = NeuralNetwork.sigmoid(self, self.outputUnits[x].output)
        #
        
    #
    
    def performCleanUp(self):
        
        itemsToClean = [self.inputUnits, self.hiddenUnits, self.outputUnits]
        
        for h in range(0, len(itemsToClean)):
            for x in range(0, len(itemsToClean[h])):
                itemsToClean[h][x].error = 0
                itemsToClean[h][x].output = 0
                itemsToClean[h][x].inputs = [0] * len(itemsToClean[h][x].inputs)
            #
        #
        
    #
    
    def trainOnExample(self, inputVector, outputVector):
        
        NeuralNetwork.doForwardPropagation(self, inputVector)
        hiddenUnits = [Neuron(self.hiddenUnits[x].edgeWeights, self.hiddenUnits[x].inputs, self.hiddenUnits[x].error, self.hiddenUnits[x].output) for x in range(len(self.hiddenUnits))]
        outputUnits = [Neuron(self.outputUnits[x].edgeWeights, self.outputUnits[x].inputs, self.outputUnits[x].error, self.outputUnits[x].output) for x in range(len(self.outputUnits))]
        array = [hiddenUnits, outputUnits]
        NeuralNetwork.doBackwardPropagation(self, outputVector)
        NeuralNetwork.performCleanUp(self)
        
        return array
        
    #
    
#
