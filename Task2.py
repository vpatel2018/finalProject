import neuralNetwork as nn
import math
import os

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
    
   return sseFile
    
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

def appendListContentsToFile(fileName, array, numDigitsAfterDecimalPlace):
 
    file = open(fileName, 'a')
    string = ""
    for z in array:
        string += (str(z)[0: numDigitsAfterDecimalPlace + str(z).find('.') + 1] + ',')   
    #
    string = string[0: len(string) - 1] + '\n'
    file.write(string)
    file.close()
    
#

def appendListContentsToFile2(fileName, array):
    
    file = open(fileName, 'a')
    string = ""
    for z in array:
        string += (str(z) + ',')   
    #
    string = string[0: len(string) - 1] + '\n'
    file.write(string)
    file.close()
    
#

if __name__ == "__main__":
   
   numInputUnits = 8
   numHiddenUnits = 3
   numOutputUnits = 8
   lowerLimitForRandNums = -0.09999999999999999
   upperLimitForRandNums = 0.09999999999999999
   learningRate = 0.3
   epochs = 5000
   network = nn.NeuralNetwork(numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate)
   vectors = [[0] * 8 for x in range(8)]

   for x in range(0, 8):
       vectors[x][x] = 1
   #
   
   trainingSet = [[vectors[x].copy(), vectors[x].copy()] for x in range(8)]
   sseFile = createSSEFile()
   hueFiles = createHiddenUnitEncodingFiles()
   numDigitsAfterDecimalPlace = 4
    
   print('D = {', end='')

   for x in range(0, len(trainingSet) - 1):
       print(trainingSet[x])    
   #

   print(trainingSet[len(trainingSet) - 1], end='}')
   print()
   
   os.system('rm D2/* > /dev/null 2>&1')
   os.system('rm -d D2 > /dev/null 2>&1')
   os.system('mkdir D2 > /dev/null 2>&1')
   
   for x in range(0, epochs):
       
       outputUnitOutputsForEpoch = []
       
       for y in range(0, len(trainingSet)):
           array = network.trainOnExample(trainingSet[y][0], trainingSet[y][1])         
           appendListContentsToFile2(hueFiles[y], array[0])
           outputUnitOutputsForEpoch.append(array[1])
       #
       
       errorsForEpoch = []
       numOfCols = len(outputUnitOutputsForEpoch[0])
       numOfRows = len(outputUnitOutputsForEpoch)
       
       for y in range(0, numOfCols):
           predictedValues = [outputUnitOutputsForEpoch[z][y] for z in range(numOfRows)]
           expectedValues = [vectors[z][y] for z in range(numOfRows)]
           sumOfSquareErr = computeSumOfSquareError(predictedValues, expectedValues)
           errorsForEpoch.append(sumOfSquareErr)
       #
    
       appendListContentsToFile(sseFile, errorsForEpoch, numDigitsAfterDecimalPlace)
       
   #
   
   os.system('mv SumOfSquaredErrors.csv D2/SumOfSquaredErrors.csv')
   
   for x in hueFiles:
       command = 'mv ' + x + ' D2/' + x
       os.system(command)
   #

# 
