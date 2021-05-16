# CSC 426 Final Project: Neural Networks Implementation and Analysis
# Authors: Vihan Patel, Lana Abdelmohsen, Luke Kurlandski, Amulya Badineni

# Content Guide and High-level Description of Code
1. neuralNetwork.py 
  - Contains our neural network class to create a neural network.
  - Contains a neuron class used to create a node for a neural network.
  - Contains functions for updating hidden unit errors and output unit errors. 
    - The functions use target values as a parameter (list of values that will be compared with the outputs of all the output units in a neural network) and calculate the squared error. 
  - Contains a function to calculate weight updates by going through every input unit in the neural network and updating the weights of edges connecting an input unit to all hidden units in a neural network.
  - Contains a backpropagation neural network learning algorithm
  - Contains a functions to update weights, and outputs for hidden and output units.
  - Contains a sigmoid method that calculates the sigmoid function which uses outputValue as a parameter which is the output of a unit in a neural network and the method outputs a decimal value between 0 and 1. 
  - Contains a method for forward propagation which uses the inputVector as a parameter which is a list of values that will be fed to the input units. It's functionality is that it stores input values in input units and computes outputs for all hidden and output units. 
  - Contains a training function that helps the neural network learn from a training example and it uses inputVector as a parameter and outputVector as parameter (which is a list of values that will be compared with the outputs of all output units).
2. Task2.py 
  - produces a folder called D2, uses neuralNetwork.py.
  - Contains a function for creating the sum of squared errors file.
  - Contains a function that creates eight hidden unit encoding files and returns a list containing names of the hidden unit encoding files created. 
  - Contains a function that writes a new line to a csv file and takes into account the number of digits that should come after a decimal place for a value that is written to a csv file. 
  - Contains a function that computes the sum of squared error
3. D2  
  - folder containing files required for D2, produced by Task2.py
4. D3 
  - folder containing plots required for D3
5. Task4Table.csv
  - a single summary hidden representations file that has eight lines in total that highlights the three values on the last line of the files in D2, after the 5,000th epoch of learning
6. Task 4 Analysis (WHAT IS NAME OF FILE???)
  - An analysis of the summary of the eight lines of Task4Table.csv

------------------------------------------------------------------------------------------------------------

# STEPS FOR RUNNING SOURCE CODE

1. Figure out the absolute path for a folder called finalProject. 
2. Type 'cd @', where @ represents the absolute path for a folder called finalProject. Press the return key.
3. Type 'module add python' and press the return key.
4. Type 'python3 Task2.py' and press the return key.
