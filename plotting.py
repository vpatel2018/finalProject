"""
Create the required plots.

To use via the command line
------------------------------------------------------------------------

	> python3 plotting.py

Notes
------------------------------------------------------------------------
	Requires third party library pandas be installed to produce the
		plots and numpy library to perform the testing.
"""

import os
import random
import sys

try:
	import numpy as np
except ModuleNotFoundError:
	print("Warning: 'numpy' library not installed for plotting.py")
except Exception:
	print("Warning: some strange business is occuring with 'numpy'.")

try:
	import pandas as pd
except ModuleNotFoundError:
	print("Warning: 'pandas' library not installed for plotting.py")
except Exception:
	print("Warning: some strange business is occuring with 'pandas'.")

def sum_of_squared_errors(path_in, path_out):
	"""
	Produce the plots for the sum of squared errors, as in task 3.

	Arguments
	--------------------------------------------------------------------
		path_in : str : the correctly formatted input path to csv file
		path_out : str : the output png path for the plot

	Notes
	--------------------------------------------------------------------
		Expects formatted csv file with eight unnamed columns:
			SSE output unit1, SSE output unit2, ... , SSE output unit8
	"""

	colnames = [f'output unit {i}' for i in range(1,9)]
	df = pd.read_csv(path_in, header=None, names=colnames)
	df['epoch'] = df.index

	ax = df.plot.line(x='epoch', y=colnames)

	ax.set_xlabel('epoch')
	ax.set_ylabel('sum of squared errors')
	ax.set_title('Sum of Squared Errors for Each Output Unit')
	ax.figure.savefig(path_out)

def hidden_unit_encodings(path_in, path_out, name):
	"""
	Produce the plots for the hidden unit encodings, as in task 3.

	Arguments
	--------------------------------------------------------------------
		path_in : str : the correctly formatted input path to csv file
		path_out : str : the output png path for the plot
		name : str : the name of this unit, for ex '00000001'

	Notes
	--------------------------------------------------------------------
		Expects formatted csv file with three unnamed columns:
			HiddenUnit1Encoding, HiddenUnit2Encoding, HiddenUnit3Encoding
	"""

	colnames = [f'hidden unit encoding {i}' for i in range(1,4)]
	df = pd.read_csv(path_in, header=None, names=colnames)
	df['epoch'] = df.index

	ax = df.plot.line(x='epoch', y=colnames)

	ax.set_xlabel('epoch')
	ax.set_ylabel('values emitted by hidden units')
	ax.set_title(f'Hidden Unit Encoding for Input {name}')
	ax.figure.savefig(path_out)

def test_sum_of_squared_errors():
	"""
	Testing function.
	"""

	path_in = 'test_in_sum_of_squared_errors.csv'
	path_out = 'test_out_sum_of_squared_errors.png'

	df = pd.DataFrame(np.random.rand(5000, 8))
	df.to_csv(path_in, header=False, index=False)

	sum_of_squared_errors(path_in=path_in, path_out=path_out)

def test_hidden_unit_encodings():
	"""
	Testing function.
	"""

	path_in = 'test_in_hidden_unit_encodings.csv'
	path_out = 'test_out_hidden_unit_encodings.png'
	name = 'name'

	df = pd.DataFrame(np.random.rand(5000, 3))
	df.to_csv(path_in, header=False, index=False)

	hidden_unit_encodings(path_in=path_in, path_out=path_out, name=name)

def main():
	"""
	Produce the required plots.
	"""
	
	sum_of_squared_errors(
		path_in='D2.SumOfSquaredErrors.csv', 
		path_out='D2.SumOfSquaredErrors.png'
	)

	names = ['00000001','00000010','00000100','00001000',
			'00010000','00100000','01000000','10000000']

	for name in names:
		hidden_unit_encodings(
			path_in=f'D3.HiddenUnitEncoding.{name}.csv', 
			path_out=f'D3.HiddenUnitEncoding.{name}.png', 
			name=name
		)

if __name__ == "__main__":
	test_sum_of_squared_errors()
	test_hidden_unit_encodings()
	
	# FIXME: remove testing function, run main instead
	#main()