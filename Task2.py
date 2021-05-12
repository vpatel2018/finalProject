import neuralNetwork as nn
import math

#TODO: Do additional debugging if needed, check if requirements for task 2 are met, write comments

def createSSEFile():
    
   sseFile = 'SumOfSquaredErrors.csv'
   file = open(sseFile, 'w')
   string = ""
   for x in range(0, 8):
       string += ('SSE_output_unit' + str(x + 1) + ',')   
   #
   string = string[0:len(string) - 1] + '\n'
   file.write(string)
   file.close()
    
#

def createHiddenUnitEncodingFiles():
    
   vectors = [[0] * 8 for x in range(8)]

   for x in range(0, 8):
       vectors[x][x] = 1
   #
    
   hiddenUnitEncodingFiles = []
   
   for x in range(0, 8):
       word = ''
       for y in range(0, len(vectors[x])):
           vectors[x][y] = str(vectors[x][y])
       #
       word = word.join(vectors[x])
       title = 'HiddenUnitEncoding_' + word + '.csv'
       hiddenUnitEncodingFiles.append(title)
       file = open(title, 'w')
       file.write('HiddenUnit1Encoding,HiddenUnit2Encoding,HiddenUnit3Encoding\n')
       file.close()
   #
   
   return hiddenUnitEncodingFiles

#

def computeSumOfSquareError(predictedValues, expectedValues):
    
    sumOfSquareErrors = 0
    
    for x in range(0, len(predictedValues)):
        value1 = predictedValues[x]
        value2 = expectedValues[x]
        difference = value2 - value1
        sumOfSquareErrors = sumOfSquareErrors + math.pow(difference, 2)
    #
    
    return sumOfSquareErrors
       
#

if __name__ == "__main__":
   
   numInputUnits = 8
   numHiddenUnits = 3
   numOutputUnits = 8
   lowerLimitForRandNums = -0.1
   upperLimitForRandNums = 0.1
   learningRate = 0.3
   epochs = 5000
   network = nn.NeuralNetwork(numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate)
   vectors = [[0] * 8 for x in range(8)]

   for x in range(0, 8):
       vectors[x][x] = 1
   #
   
   trainingSet = [[vectors[x].copy(), vectors[x].copy()] for x in range(8)]
   createSSEFile()
   hueFiles = createHiddenUnitEncodingFiles()
   
   for x in range(0, epochs):
       
       outputUnitOutputsForEpoch = []
       
       for y in range(0, len(trainingSet)):

           inputForNN = trainingSet[y][0]
           expectedOutput = trainingSet[y][1]
           array = network.trainOnExample(inputForNN, expectedOutput)
           hiddenUnitOutputs = array[0]

           file = open(hueFiles[y], 'a')
           string = ""
           for z in hiddenUnitOutputs:
               string += (str(z) + ',')   
           #
           string = string[0: len(string) - 1] + '\n'
           file.write(string)
           file.close()

           outputUnitOutputs = array[1]
           outputUnitOutputsForEpoch.append(outputUnitOutputs)
            
       #
       
       errorsForEpoch = []
       
       for y in range(0, len(outputUnitOutputsForEpoch[0])):
           predictedValues = []
           expectedValues = []
           for z in range(0, len(outputUnitOutputsForEpoch)):
               value = outputUnitOutputsForEpoch[z][y]
               predictedValues.append(value)
               value = vectors[z][y]
               expectedValues.append(value)
           #
           sumOfSquareErr = computeSumOfSquareError(predictedValues, expectedValues)
           errorsForEpoch.append(sumOfSquareErr)
       #
       
       sseFile = 'SumOfSquaredErrors.csv'
       file = open(sseFile, 'a')
       string = ""
       for y in errorsForEpoch:
           string += (str(y) + ',')
       #
       string = string[0:len(string) - 1] + '\n'
       file.write(string)
       file.close()
       
   #

#
