import neuralNetwork as nn
import math

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
   
   trainingSet = [[vectors[x], vectors[x]] for x in range(8)]
   sseFile = 'SumOfSquaredErrors.csv'
   file = open(sseFile, 'w')
   file.close()
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
       file.close()
   #
   
   #write code to deal with hidden unit encodings
   #write code to deal with sum of squared errors
   
#
