import random
import math

#TODO: edit, revise, or add more comments

#used to create a node for neural network
class Neuron:
    def __init__(self, edgeWeights=[], inputs=[], error=0, output=0):
        self.edgeWeights = edgeWeights #contains weights of edges that connect a unit to multiple other units
        self.inputs = inputs #contains value that will go from an input unit to every hidden unit or from a hidden unit to every output unit
        self.error = error #error for a unit
        self.output = output #value that a unit outputs
    # 
#

class NeuralNetwork:
    
    #*********************************************************************************************************#
     
    #this method is used to create a neural network
    def __init__(self, numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate):
        
        '''
        numInputUnits = number of input units in a neural network
        numHiddenUnits = number of hidden units in a neural network
        numOutputUnits = number of output units in a neural network
        lowerLimitForRandNums = minimum value for a random number
        upperLimitForRandNums = maximum value for a random number
        learningRate = a float representing a learning rate
        
        NOTE: all weights for the neural network will be initialized to a number >= lowerLimitForRandNums and number <= upperLimitForRandNums 
        '''
        
        self.outputUnits = [Neuron() for x in range(numOutputUnits)]
        self.hiddenUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numOutputUnits)], [0 for y in range(numOutputUnits)]) for x in range(numHiddenUnits)]
        self.inputUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numHiddenUnits)], [0 for y in range(numHiddenUnits)]) for x in range(numInputUnits)]
        self.learningRate = learningRate

    #
    
    #*********************************************************************************************************#
    
    #computes the error of an output unit and stores it in the output unit for every output unit in a neural network
    def updateOutputUnitErrors(self, targetValues):
        
        '''
        targetValues = a list of values that will be compared with outputs of all the output units in a neural network
        '''

        for x in range(0, len(self.outputUnits)):
            output = self.outputUnits[x].output
            target = targetValues[x]
            self.outputUnits[x].error = (output) * (1 - output) * (target - output) 
        #

    #
        
    #*********************************************************************************************************#

    #computes the error of a hidden unit and stores it in the hidden unit for every hidden unit in a neural network
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
        targetValues = a list of values that will be compared with outputs of all the output units in a neural network
        '''
        
        NeuralNetwork.updateOutputUnitErrors(self, targetValues) 
        NeuralNetwork.updateHiddenUnitErrors(self)
        NeuralNetwork.updateNetworkWeights(self)
        
    #
    
    #*********************************************************************************************************#
    
    #computes the output of a hidden unit and stores it in the hidden unit
    def updateOutputForHiddenUnit(self, index):
        
        '''
        -> index is a number >= 0
        -> Let's say we want to compute an output and store it in hidden unit #3. Then, index has to equal 2.
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
    
    #computes the output of an output unit and stores it in the output unit
    def updateOutputForOutputUnit(self, index):
        
        '''
        -> index is a number >= 0
        -> Let's say we want to compute an output and store it in output unit #3. Then, index has to equal 2.
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
        input: outputValue is a number representing the output of a unit in a neural network
        
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
    
    #sets the input values, output and error of each unit in a neural network to zero
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
    
    #used to help a neural network learn from a training example
    def trainOnExample(self, inputVector, outputVector):
        
        '''
        input: inputVector represents a list of numbers that will be fed to all the input units of a neural network, 
        outputVector represents a list of values that should be outputted by all output units in a neural network
        
        output: a list of the format [hiddenUnitOutputs, outputUnitOutputs], where hiddenUnitOutputs is a
        list of outputs for all hidden units in a neural network and outputUnitOutputs is a list of outputs
        for all output units in a neural network
        '''
        
        NeuralNetwork.doForwardPropagation(self, inputVector) 
        numHiddenUnits = len(self.hiddenUnits) 
        numOutputUnits = len(self.outputUnits) 
        hiddenUnitOutputs = [self.hiddenUnits[x].output for x in range(numHiddenUnits)] 
        outputUnitOutputs = [self.outputUnits[x].output for x in range(numOutputUnits)] 
        array = [hiddenUnitOutputs, outputUnitOutputs] 
        NeuralNetwork.doBackwardPropagation(self, outputVector) 
        NeuralNetwork.performCleanUp(self) 
        
        return array
        
    #
    
    #*********************************************************************************************************#
    
#            
