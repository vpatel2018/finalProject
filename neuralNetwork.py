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
