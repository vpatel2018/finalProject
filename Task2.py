# Authors: Lana Abdelmohsen, Amulya Badineni, Luke Kurlandski, Vihan Patel

import neuralNetwork as nn
import math
import os

#creates sum of squared errors file required for task 2
#returns the name of the sum of squared errors file that is created
def createSSEFile():
    
   #create sum of squared errors file called SumOfSquaredErrors.csv
   sseFile = 'SumOfSquaredErrors.csv'
   file = open(sseFile, 'w')
   
   #write the columns SSE_output_unit1, ..., SSE_output_unit8 in SumOfSquaredErrors.csv
   string = ""
   for x in range(0, 8):
       string += ('SSE_output_unit' + str(x + 1) + ',')   
  
   string = string[0:len(string) - 1] + '\n'
   file.write(string)
   file.close()
    
   return sseFile
    
#creates eight hidden unit encoding files required for task 2
#returns a list containing names of the hidden unit encoding files created
def createHiddenUnitEncodingFiles():
   
   #represents list of all input vectors for task 2
   vectors = [[0] * 8 for x in range(8)]

   for x in range(0, 8):
       vectors[x][x] = 1
    
   #list containing names of the hidden unit encoding files created
   hiddenUnitEncodingFiles = []
   
   for x in range(0, 8):
       
       word = ''
       for y in range(0, len(vectors[x])):
           vectors[x][y] = str(vectors[x][y])
     
       word = word.join(vectors[x])
       
       #create hidden unit encoding file
       title = 'HiddenUnitEncoding_' + word + '.csv'
       hiddenUnitEncodingFiles.append(title)
       file = open(title, 'w')
       
       #write the columns HiddenUnit1Encoding, ..., HiddenUnit3Encoding in hidden unit encoding file
       file.write('HiddenUnit1Encoding,HiddenUnit2Encoding,HiddenUnit3Encoding\n')
       file.close()
       

   return hiddenUnitEncodingFiles

#used to compute sum of square error
def computeSumOfSquareError(predictedValues, expectedValues):
    
    '''
    predictedValues -> list of predicted values
    expectedValues -> list of expected values
    '''
    
    sumOfSquareErrors = 0
    
    #take summation of differences between predicted and expected values
    for x in range(0, len(predictedValues)):
        sumOfSquareErrors += math.pow(expectedValues[x] - predictedValues[x], 2)
    
    
    return sumOfSquareErrors
       
#writes a new line to a csv file
#takes into account the number of digits that should come after a decimal place for a value that is written to a csv file
def appendListContentsToFile(fileName, array, numDigitsAfterDecimalPlace):
    
    '''
    fileName -> represents name of csv file we are going to write to
    array -> represents list of values we have to include when writing a new line in a csv file
    numDigitsAfterDecimalPlace -> represents number of digits that should come after a decimal place for a value
    '''
    
    #open a csv file
    file = open(fileName, 'a')
    
    #write a new line in a csv file
    string = ""
    for z in array:
        if(numDigitsAfterDecimalPlace == 0):
          string += (str(z)[0: str(z).find('.')] + ',')
        else:
          string += (str(z)[0: numDigitsAfterDecimalPlace + str(z).find('.') + 1] + ',')      
        
    
    string = string[0: len(string) - 1] + '\n'
    file.write(string)
    file.close()
    
#writes a new line to a csv file
#does not take into account # of digits that should come after a decimal place for a value that is written to a csv file
def appendListContentsToFile2(fileName, array):
    
    '''
    fileName -> represents name of csv file we are going to write to
    array -> represents list of values we have to include when writing a new line in a csv file
    '''
    
    #open a csv file
    file = open(fileName, 'a')
    
    #write a new line in a csv file
    string = ""
    for z in array:
        string += (str(z) + ',')   
    
    string = string[0: len(string) - 1] + '\n'
    file.write(string)
    file.close()
    
if __name__ == "__main__":
   
   #represents number of input units for a neural network
   numInputUnits = 8

   #represents number of hidden units for a neural network
   numHiddenUnits = 3

   #represents number of output units for a neural network
   numOutputUnits = 8

   #minimum value for the initial values of neural network weights
   lowerLimitForRandNums = -0.1

   #maximum value for the initial values of neural network weights
   upperLimitForRandNums = 0.1

   #represents learning rate for neural network
   learningRate = 0.3

   #represents number of epochs of learning that have to be done
   epochs = 5000

   #represents neural network we will use for learning
   network = nn.NeuralNetwork(numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate)

   #represents list of all input vectors for task 2
   vectors = [[0] * 8 for x in range(8)]

   for x in range(0, 8):
       vectors[x][x] = 1
   
   
   #represents training set
   trainingSet = [[vectors[x].copy(), vectors[x].copy()] for x in range(8)]
    
   #represents name of sum of squared errors file that will be produced
   sseFile = createSSEFile()
    
   #contains names of hidden unit encoding files that will be produced
   hueFiles = createHiddenUnitEncodingFiles()

   #print information about training set, neural network architecture used, and learning rate for neural network
   print()
   print('NETWORK ARCHITECTURE =', numInputUnits, 'x', numHiddenUnits, 'x', numOutputUnits)
   print('LEARNING RATE =', learningRate)
   print()
   print('D = {', end='')
   for x in range(0, len(trainingSet) - 1):
       print(trainingSet[x])    
   
   print(trainingSet[len(trainingSet) - 1], end='}')
   print()
   print()
   
   #create a new folder called D2
   os.system('rm D2/* > /dev/null 2>&1')
   os.system('rm -d D2 > /dev/null 2>&1')
   os.system('mkdir D2 > /dev/null 2>&1')
   
   #perform 5000 epochs of learning
   for x in range(0, epochs):
       
       #used to store the outputs of all output units for each training example
       outputUnitOutputsForEpoch = []
 
       #write outputs of hidden units to hidden unit encoding files
       for y in range(0, len(trainingSet)):
           array = network.trainOnExample(trainingSet[y][0], trainingSet[y][1])         
           appendListContentsToFile2(hueFiles[y], array[0])
           outputUnitOutputsForEpoch.append(array[1])
       
       
       #obtain values needed to write a line in SumOfSquaredErrors.csv
       errorsForEpoch = []
       numOfCols = len(outputUnitOutputsForEpoch[0])
       numOfRows = len(outputUnitOutputsForEpoch)
       
       for y in range(0, numOfCols):
           predictedValues = [outputUnitOutputsForEpoch[z][y] for z in range(numOfRows)]
           expectedValues = [vectors[z][y] for z in range(numOfRows)]
           sumOfSquareErr = computeSumOfSquareError(predictedValues, expectedValues)
           errorsForEpoch.append(sumOfSquareErr)
       
       
       #write a line in SumOfSquaredErrors.csv
       #make sure each value for a line in SumOfSquaredErrors.csv has at most 4 digits after a decimal place
       numDigitsAfterDecimalPlace = 4
       appendListContentsToFile(sseFile, errorsForEpoch, numDigitsAfterDecimalPlace)
       
   #make sure SumOfSquaredErrors.csv and all hidden unit encoding files produced are placed in D2
   os.system('mv SumOfSquaredErrors.csv D2/SumOfSquaredErrors.csv')
   
   for x in hueFiles:
       command = 'mv ' + x + ' D2/' + x
       os.system(command)
   


