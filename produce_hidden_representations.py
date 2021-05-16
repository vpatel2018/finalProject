"""
Produce the HiddenRepresentationsFile.csv and HiddenRepresentationsFileRounded.csv.
"""

import pandas as pd

def main():

	names = ['00000001','00000010','00000100','00001000',
		'00010000','00100000','01000000','10000000']

	cols = ['input', 'unit1', 'unit2', 'unit3', 'output']

	df_output = pd.DataFrame(columns=cols)

	for name in names:
		df_input = pd.read_csv(f'D2/HiddenUnitEncoding_{name}.csv')
		tail = df_input.tail(1).to_dict(orient='list')
		data = {
			'input' : name, 'output' : name,
			'unit1' : tail['HiddenUnit1Encoding'][0],
			'unit2' : tail['HiddenUnit2Encoding'][0],
			'unit3' : tail['HiddenUnit3Encoding'][0]
		}
		df_output = df_output.append(data, ignore_index=True)

	df_output.to_csv('HiddenRepresentationsFile.csv', index=False)
	(df_output
		.round(0)
		.astype({'unit1' : 'int32', 'unit2' : 'int32', 'unit3' : 'int32'})
		.to_csv('HiddenRepresentationsFileRounded.csv', index=False)
	)

if __name__ == "__main__":
	main()
