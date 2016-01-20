# Rikk's exploratory gubbins
import pandas as pd
import numpy as np
import helper.data as hd
import helper.plot as plot
import helper.constants as C


district_list = pd.unique(data['PdDistrict'].ravel())


for district in district_list:
	print(district)
	print(data['Category'].value_counts())
	print("\n###############\n\n")