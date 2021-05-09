import random

#you can add additional fields if needed
#this is used to create a neural network node
class Neuron:
    def __init__(self, nextNodes=[], edgeWeights=[], error=0):
        self.nextNodes = nextNodes
        self.edgeWeights = edgeWeights
        self.error = error
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
    
    #used for testing purposes
    def printNeuralNetwork(self):
        
        print('*********************************')
        
        print('Checking list of input units')
        print('# of hidden units = ', len(self.hiddenUnits))
        
        for x in self.inputUnits:
            #tests if all nodes in self.inputUnits has node.nextNodes equal to hidden units
            #tests to see if the number of weights is equal to number of hidden units
            #tests to see if weights are all unique and random
            print(x.nextNodes == self.hiddenUnits, len(x.edgeWeights) == len(self.hiddenUnits), ', initial edge weights =', x.edgeWeights)
        #
        
        print('*********************************')
        
        print('Checking list of hidden units')
        print('# of output units = ', len(self.outputUnits))
        
        for x in self.hiddenUnits:
            #tests if all nodes in self.hiddenUnits has node.nextNodes equal to output units
            #tests to see if the number of weights is equal to number of output units
            #tests to see if weights are all unique and random
            print(x.nextNodes == self.outputUnits, len(x.edgeWeights) == len(self.outputUnits), ', initial edge weights = ', x.edgeWeights)
        #        
        
        print('*********************************')
        
    #

    def getErrorOfOutputUnit():

        pass

    #

    def getErrorOfHiddenUnit():

        pass

    #
    
#

numInputUnits = 3
numHiddenUnits = 5
numOutputUnits = 6
lowerLimit = 1
upperLimit = 2
network = NeuralNetwork(numInputUnits, numHiddenUnits, numOutputUnits, lowerLimit, upperLimit)
network.printNeuralNetwork()
