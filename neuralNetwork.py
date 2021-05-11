import random
import math

#TODO: write multi-line and single line comments

class Neuron:
    def __init__(self, edgeWeights=[], inputs=[], error=0, output=0):
        self.edgeWeights = edgeWeights
        self.inputs = inputs
        self.error = error
        self.output = output
    # 
#

class NeuralNetwork:
    
    #*********************************************************************************************************#
    
    def __init__(self, numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate):
        
        self.outputUnits = [Neuron() for x in range(numOutputUnits)]
        self.hiddenUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numOutputUnits)], [0 for y in range(numOutputUnits)]) for x in range(numHiddenUnits)]
        self.inputUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numHiddenUnits)], [0 for y in range(numHiddenUnits)]) for x in range(numInputUnits)]
        self.learningRate = learningRate

    #
    
    #*********************************************************************************************************#
    
    def updateOutputUnitErrors(self, targetValues):

        for x in range(0, len(self.outputUnits)):
            output = self.outputUnits[x].output
            target = targetValues[x]
            self.outputUnits[x].error = (output) * (1 - output) * (target - output) 
        #

    #
    
    #*********************************************************************************************************#
    
    def getSummForHiddenUnit(self, hiddenUnit):
        
        total = 0
        edgeWeights = hiddenUnit.edgeWeights
        numOutputUnits = len(self.outputUnits)
        outputUnitErrors = [self.outputUnits[h].error for h in range(numOutputUnits)]
        
        for y in range(0, len(edgeWeights)):
            product = edgeWeights[y] * outputUnitErrors[y]
            total += product
        #
        
        return total
        
    #
    
    #*********************************************************************************************************#

    def updateHiddenUnitErrors(self):

        for x in range(0, len(self.hiddenUnits)):
            total = NeuralNetwork.getSummForHiddenUnit(self, self.hiddenUnits[x])
            output = self.hiddenUnits[x].output
            self.hiddenUnits[x].error = output * (1 - output) * total
        #   

    #
    
    #*********************************************************************************************************#
    
    def updateNetworkWeights(self):
        
        for x in range(0, len(self.inputUnits)): 
            inputUnit = self.inputUnits[x] 
            for y in range(0, len(self.hiddenUnits)): 
                delta = self.hiddenUnits[y].error 
                inputValue = inputUnit.inputs[y] 
                change = delta * inputValue * self.learningRate 
                self.inputUnits[x].edgeWeights[y] += change 
            #
        #
        
        for x in range(0, len(self.hiddenUnits)): 
            hiddenUnit = self.hiddenUnits[x] 
            for y in range(0, len(self.outputUnits)): 
                delta = self.outputUnits[y].error 
                inputValue = hiddenUnit.inputs[y] 
                change = delta * inputValue * self.learningRate 
                self.hiddenUnits[x].edgeWeights[y] += change 
            #
        #
        
    #
    
    #*********************************************************************************************************#
    
    def doBackwardPropagation(self, targetValues):
        
        NeuralNetwork.updateOutputUnitErrors(self, targetValues) 
        NeuralNetwork.updateHiddenUnitErrors(self)
        NeuralNetwork.updateNetworkWeights(self)
        
    #
    
    #*********************************************************************************************************#
    
    def updateOutputForHiddenUnit(self, index):
        
        total = 0
        
        for x in range(0, len(self.inputUnits)): 
            inputUnit = self.inputUnits[x] 
            inputValue = inputUnit.inputs[index] 
            weight = inputUnit.edgeWeights[index] 
            product = inputValue * weight 
            total += product 
        #
        
        self.hiddenUnits[index].output = total 
           
    #
    
    #*********************************************************************************************************#
    
    def updateOutputForOutputUnit(self, index):
        
        total = 0
        
        for x in range(0, len(self.hiddenUnits)): 
            hiddenUnit = self.hiddenUnits[x] 
            inputValue = hiddenUnit.inputs[index] 
            weight = hiddenUnit.edgeWeights[index] 
            product = inputValue * weight 
            total += product 
        #
        
        self.outputUnits[index].output = total 
        
    #
    
    #*********************************************************************************************************#
    
    def sigmoid(self, output):
        
        negatedOutput = -1 * output
        denominator = 1 + math.pow(math.e, negatedOutput)
        result = (1.0) / (1.0 * denominator)
            
        return result
               
    #
    
    #*********************************************************************************************************#
   
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
    
    #*********************************************************************************************************#
    
    def performCleanUp(self):
        
        itemsToClean = [self.inputUnits, self.hiddenUnits, self.outputUnits] 
        
        for h in range(0, len(itemsToClean)): 
            item = itemsToClean[h] 
            for x in range(0, len(item)): 
                item[x].error = 0 
                item[x].output = 0 
                item[x].inputs = [0] * len(item[x].inputs) 
            #
        #
        
    #
    
    #*********************************************************************************************************#
    
    #TODO: test this function
    def trainOnExample(self, inputVector, outputVector):
        
        NeuralNetwork.doForwardPropagation(self, inputVector)
        hiddenUnits = [hiddenUnits[x].copy() for x in range(len(self.hiddenUnits))]
        outputUnits = [outputUnits[x].copy() for x in range(len(self.outputUnits))]
        array = [hiddenUnits, outputUnits]
        NeuralNetwork.doBackwardPropagation(self, outputVector)
        NeuralNetwork.performCleanUp(self)
        
        return array
        
    #
    
#
