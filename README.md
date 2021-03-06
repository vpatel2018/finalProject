# CSC 426 Final Project: Neural Networks Implementation and Analysis
# Authors: Vihan Patel, Lana Abdelmohsen, Luke Kurlandski, Amulya Badineni

# Content Guide and High-level Description of Code
1. neuralNetwork.py 
  - Contains our neural network class to create a neural network.
  - Contains a neuron class used to create a node for a neural network.
  - Contains functions for updating hidden unit errors and output unit errors. 
    - The function for output unit errors uses target values as a parameter (list of values that will be compared with the outputs of all the output units in a neural network) and calculate the squared error. 
  - Contains a function to calculate weight updates 
    - By going through every input unit in the neural network and updating the weights of edges connecting an input unit to all hidden units in a neural network.
    - By going through every hidden unit in a neural network and updating the weights of edges connecting a hidden unit to all output units in a neural network
  - Contains a backpropagation neural network learning algorithm that calls the output unit error function, "updateHiddenUnitErrors" and "updateOutputUnitErrors" and "updateNetworkWeights" functions to update weights and errors. 
  - Contains outputs for hidden and output units.
    - To update the output for a hidden unit the function uses the weights of edges connecting all the input units to a hidden unit and uses the values that all input units will transfer to a hidden unit during forward propagation
  - Contains a sigmoid method that calculates the sigmoid function which uses "outputValue" as a parameter which is the output of a unit in a neural network and the method outputs a decimal value between 0 and 1. 
  - Contains a method for forward propagation which uses the "inputVector" as a parameter which is a list of values that will be fed to the input units. Its functionality is that it stores input values in input units and computes outputs for all hidden and output units. 
  - Contains a training function that helps the neural network learn from a training example and it uses "inputVector" as a parameter and "outputVector" as a parameter (which is a list of values that will be compared with the outputs of all output units).
    - It calls the functions for forward and backward propagation
2. Task2.py 
  - Produces a folder called D2, uses neuralNetwork.py.
  - Defines the learning rate, epochs to run, the upper and lower bounds of the weights and the multi-layer 8x3x8 neural network
  - Contains a function for creating the sum of squared errors file.
  - Contains a function that creates eight hidden unit encoding files and returns a list containing names of the hidden unit encoding files created. 
  - Contains a function that writes a new line to a csv file and takes into account the number of digits that should come after a decimal place for a value that is written to a csv file. 
  - Contains a function that computes the sum of squared error
3. D2  
  - Folder containing files required for D2, produced by Task2.py
4. D3 
  - Folder containing plots required for D3
5. HiddenRepresentationsFile.csv
  - A single summary hidden representations file that has eight lines in total that highlight the three values on the last line of the files in D2, after the 5,000th epoch of learning
6. D5.txt
  - The reflection in responses to: 
    - What was easy about this assignment?
    - What was challenging about this assignment, or parts that you couldn???t get working correctly?
    - What did you like about this assignment?
    - What did you dislike about this assignment?
    - How did your team function, including details such as what each team member contributed, how the team communicated with each other, and how team software development & design was accomplished?
    - What did you learn from this assignment?
7. Analysis.pdf
  - An analysis of the summary of the eight lines of HiddenRepresentationsFile.csv for task 4
8. Task3.py
  - The code used to generate all plots, not required to run this file as plots are found in D3 folder.  
------------------------------------------------------------------------------------------------------------

Note: The graphs in D3 are based on files in D2, so you might want to look at D3 before running Task2.py. 
If you run Task2.py, files in D2 will be replaced with new files.

# STEPS FOR RUNNING SOURCE CODE

1. Figure out the absolute path for a folder called finalProject. 
2. Type 'cd @', where @ represents the absolute path for a folder called finalProject. Press the return key.
3. Type 'module add python/3.6.0' and press the return key.
4. Type 'python3 Task2.py' and press the return key.
