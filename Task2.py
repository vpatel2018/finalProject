import neuralNetwork as nn
import math

#TODO: Check if the files produced meet the requirements for task 2

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

def createHiddenUnitEncodingFiles(vectors):
    
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
   createHiddenUnitEncodingFiles(vectors)

#
