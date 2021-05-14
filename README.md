# CSC 426 Final Project: Neural Networks Implementation and Analysis
# Authors: Vihan Patel, Lana Abdelmohsen, Luke Kurlandski, Amulya Badineni

# Content Guide and High-level Description of Code
1. neuralNetwork.py 
  - Contains our neural network class to create a neural network.
  - Contains a neuron class used to create a node for a neural network.
  -Contains functions for updating hidden unit errors and output unit errors, which update the errors for the hidden or output unit in the neural network. The functions use target values as a parameter (list of values that will be compared with the outputs of all the output units in a neural network) and calculate the squared error. 
  - contains a function to calculate weight updates by going through every input unit in the neural network and updating the weights of edges connecting an input unit to all hidden units in a neural network.
  - contains backpropagation neural network learning algorithm that use target values as a paramter 
  - Contains weights that update the output for hidden and output units by using weights of edges connecting all the hidden units to an output unit and uses the values that all the hidden units will transfer to an output unit during forward propagation
  weight updates, error and output calculations. 
  - Contains a sigmoid method that calculates the sigmoid function which uses outputValue as a parameter which is the output of a unit in a neural network and the method outputs a decimal value between 0 and 1. 
  -Contains a method for forward propagation which uses the inputVector as a parameter which is a list of values that will be fed to the input units. It's functionality is that it stores input values in input units and computes outputs for all hidden and output units. 
  - Contains a training function that helps the neural network learn from a training example and it uses inputVector as a parameter which is a list of numbers that will be fed to the input units and outputVector as parameter which is a list of values that will be compared with the outputs of all output units.
2. Task2.py 
  - produces a folder called D2, uses neuralNetwork.py.
  - Contains a function for creating the sum of squared errors file.
  - Contains a function that creates eight hidden unit encoding files and returns a list containing names of the hidden unit encoding files created. It takes fileName (represents name of csv file we are going to write to), array (represents list of values we have to include when writing a new line in a csv file) and numDigitsAfterDecimalPlace (represents number of digits that should come after a decimal place for a value) as parameters.
  -Contains a function that writes a new line to a csv file and takes into account the number of digits that should come after a decimal place for a value that is written to a csv file. It takes fileName and array as parameters. 
  -Contains a function that computes the sum of squared error and uses predictedValues (list of predicted values)
    expectedValues (list of expected values).
3. D2  
  - folder containing files required for D2, produced by Task2.py
4. D3 
  - folder containing plots required for D3
5. Task4Table.py
  - a single summary HiddenRepresentationsFile that has eight lines in total that highlights the three values on the last line of the files in D2, after the 5,000th epoch of learning
 6.Task 4 Analysis
  - An analysis of the summary of the eight linesfor the hidden unit encodings
  - What observations can you make about the machine’s hidden value encodings of the input values?  

------------------------------------------------------------------------------------------------------------

# STEPS FOR RUNNING SOURCE CODE

1. Figure out the absolute path for a folder called finalProject. 
2. Type 'cd @', where @ represents the absolute path for a folder called finalProject. Press the return key.
3. Type 'module add python' and press the return key.
4. Type 'python3 Task2.py' and press the return key.
