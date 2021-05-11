import random
import math

#TODO: polish source code documentation if needed

#used to create a node for a neural network
class Neuron:
    def __init__(self, edgeWeights=[], inputs=[], error=0, output=0):
        self.edgeWeights = edgeWeights #contains weights of edges that connect a unit to multiple other units
        self.inputs = inputs #????
        self.error = error #represents error for a unit in a neural network
        self.output = output #represents value that a unit in a neural network outputs
    # 
#

class NeuralNetwork:
    
    #*********************************************************************************************************#
    
    #used to create a neural network
    def __init__(self, numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate):
        
        '''
        numInputUnits = number of input units in a neural network
        numHiddenUnits = number of hidden units in a neural network
        numOutputUnits = number of output units in a neural network
        lowerLimitForRandNums = minimum value for a random number
        upperLimitForRandNums = maximum value for a random number
        learningRate = a float representing a learning rate
        
        NOTE: all weights for a neural network will be initialized to a number >= lowerLimitForRandNums and <= upperLimitForRandNums 
        '''
        
        self.outputUnits = [Neuron() for x in range(numOutputUnits)]
        self.hiddenUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numOutputUnits)], [0 for y in range(numOutputUnits)]) for x in range(numHiddenUnits)]
        self.inputUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numHiddenUnits)], [0 for y in range(numHiddenUnits)]) for x in range(numInputUnits)]
        self.learningRate = learningRate

    #
    
    #*********************************************************************************************************#
    
    #this function is used when back propagation is being done
    #updates the error for all the output units in a neural network
    def updateOutputUnitErrors(self, targetValues):
        
        '''
        targetValues = list of values that will be compared with the outputs of 
        all the output units in a neural network
        '''

        for x in range(0, len(self.outputUnits)):
            output = self.outputUnits[x].output
            target = targetValues[x]
            self.outputUnits[x].error = (output) * (1 - output) * (target - output) 
        #

    #
        
    #*********************************************************************************************************#

    #this function is used when back propagation is being done
    #updates the error for all the hidden units in a neural network
    def updateHiddenUnitErrors(self):

        for x in range(0, len(self.hiddenUnits)):
            
            hiddenUnit = self.hiddenUnits[x]
            edgeWeights = hiddenUnit.edgeWeights
            numOutputUnits = len(self.outputUnits)
            outputUnitErrors = [self.outputUnits[h].error for h in range(numOutputUnits)]
            
            total = 0

            for y in range(0, len(edgeWeights)):
                product = edgeWeights[y] * outputUnitErrors[y]
                total += product
            #
            
            output = self.hiddenUnits[x].output
            self.hiddenUnits[x].error = output * (1 - output) * total
            
        #   

    #
    
    #*********************************************************************************************************#
    
    #updates all the weights for a neural network
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
    
    #used to perform back propagation
    def doBackwardPropagation(self, targetValues):
        
        '''
        targetValues = list of values that will be compared with the outputs of 
        all the output units in a neural network
        '''
        
        NeuralNetwork.updateOutputUnitErrors(self, targetValues) 
        NeuralNetwork.updateHiddenUnitErrors(self)
        NeuralNetwork.updateNetworkWeights(self)
        
    #
    
    #*********************************************************************************************************#
    
    #this function is used when forward propagation is being done
    #updates the output for a hidden unit
    def updateOutputForHiddenUnit(self, index):
        
        '''
        index is a number >= 0 and it is used to refer to a certain hidden unit
        '''
        
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
    
    #this function is used when forward propagation is being done
    #updates the output for an output unit
    def updateOutputForOutputUnit(self, index):
        
        '''
        index is a number >= 0 and it is used to refer to a certain output unit
        '''
        
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
    
    #serves as a sigmoid function
    def sigmoid(self, outputValue):
        
        '''
        input: outputValue = output of a unit in a neural network
        
        output: a decimal between 0 and 1
        '''
        
        negatedOutput = -1 * outputValue
        denominator = 1 + math.pow(math.e, negatedOutput)
        result = (1.0) / (1.0 * denominator)
            
        return result
               
    #
    
    #*********************************************************************************************************#
   
    #used to perform forward propagation
    def doForwardPropagation(self, inputVector):
        
        '''
        inputVector = list of values that will be fed to the input units
        '''
        
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
    
    #used to help a neural network learn from a training example
    def trainOnExample(self, inputVector, outputVector):
        
        '''
        input: inputVector = list of numbers that will be fed to the input units, 
        outputVector = list of values that should be outputted by the output units
        
        output: a list of the format [hiddenUnitOutputs, outputUnitOutputs], where hiddenUnitOutputs is a
        list of outputs for all hidden units and outputUnitOutputs is a list of outputs for all output units
        '''
        
        NeuralNetwork.doForwardPropagation(self, inputVector) 
        numHiddenUnits = len(self.hiddenUnits) 
        numOutputUnits = len(self.outputUnits) 
        hiddenUnitOutputs = [self.hiddenUnits[x].output for x in range(numHiddenUnits)] 
        outputUnitOutputs = [self.outputUnits[x].output for x in range(numOutputUnits)] 
        array = [hiddenUnitOutputs, outputUnitOutputs] 
        NeuralNetwork.doBackwardPropagation(self, outputVector)
        itemsToClean = [self.inputUnits, self.hiddenUnits, self.outputUnits] 
        
        for h in range(0, len(itemsToClean)): 
            item = itemsToClean[h] 
            for x in range(0, len(item)): 
                item[x].error = 0 
                item[x].output = 0 
                item[x].inputs = [0] * len(item[x].inputs) 
            #
        #
        
        return array
        
    #
    
    #*********************************************************************************************************#
    
# 
