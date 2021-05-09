"""
A neural network and its associated perceptron units.
"""

from random import uniform

from NMMathFunctions import sigmoid, getErrorForOutputUnit, \
	getErrorForHiddenUnit, getWeightChange

D = [
	[1,0,0,0,0,0,0,0],
	[0,1,0,0,0,0,0,0],
	[0,0,1,0,0,0,0,0],
	[0,0,0,1,0,0,0,0],
	[0,0,0,0,1,0,0,0],
	[0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,1,0],
	[0,0,0,0,0,0,0,1],
]

class Perceptron:
	"""
	Elementary unit of the neural network.

	Parameters
	--------------------------------------------------------------------
		n_input_edges : int : the number of input edges this perceptron 
			has in the ANN. This number is one less than the length of 
			the perceptron's weights
		rnd_low : float : lower bound for random weight initialization
		rnd_up : float : upper bound for random weight initialization

	Attributes
	--------------------------------------------------------------------
		w : [float,] : list of float weights, where each weight is 
			representative of an edge entering the perceptron
	"""

	def __init__(self, n_input_edges, rnd_low=-.1, rnd_up=.1):
		self.w = [uniform(rnd_low, rnd_up) for _ in range(n_input_edges + 1)]

	def output(self, x):
		"""
		Return the output of the perceptron using the sigmoid function.

		Arguments
		----------------------------------------------------------------
			x : [float,] : input feature vector with 

		Returns
		----------------------------------------------------------------
			float : the perceptron's output for x
		"""

		return sigmoid(sum(w_i * x_i for x_i, w_i in zip(self.w, x)))

class NeuralNetwork:
	"""
	Neural network, by default configured for particular learning task.

	Parameters
	--------------------------------------------------------------------
		n_input_edges_for_input_unit : int : number of weights for 
			perceptrons in the input layer
		n_input : int : number of hidden perceptrons
		n_hidden : int : number of hidden perceptrons
		n_output : int : number of output perceptrons

	Attributes
	--------------------------------------------------------------------
		input_units : [Perceptron,] : list of input perceptrons
		hidden_units : [Perceptron,] : list of hidden perceptrons
		output_units : [Perceptron,] : list of output perceptrons
		units : [[Perceptron,],] : list of lists of output perceptrons
		n_epochs : int : number of training epochs
		eta : float : the initial learning rate
	"""

	def __init__(self, n_input_edges_for_input_unit=8, n_input=8, n_hidden=3, n_output=8):
		self.input_units = [Perceptron(n_input_edges=n_input_edges_for_input_unit) for _ in range(n_input)]
		self.hidden_units = [Perceptron(n_input_edges=n_input) for _ in range(n_hidden)]
		self.output_units = [Perceptron(n_input_edges=n_hidden) for _ in range(n_hidden)]
		self.units = [
			self.input_units,
			self.hidden_units,
			self.output_units
		]

	def fit(self, D=D, n_epochs=5000, eta=.3):
		"""
		Fit the neural network.

		Arguments
		----------------------------------------------------------------
			D : [[int,],] : training data
			n_epochs : int : number of training epochs
			eta : float : the initial learning rate
		"""

		self.n_epochs = n_epochs
		self.eta = eta
		self.back_propagation(D)

	def back_propagation(self, D):
		"""
		Fit the neural network using the backpropagation algorithm.

		Arguments
		----------------------------------------------------------------
			D : [[int,],] : training data
		
		Notes
		----------------------------------------------------------------
			Outputs sum of squared errors to D2.SumOfSquaredErrors.csv
			Outputs hidden unit encodings to eight files:
				D2.HiddenUnitEncoding00000001.csv, ... , ... ,
				D2.HiddenUnitEncodoing10000000.csv
			
		"""

		# Use functions from NMMAthFunctions
		pass



