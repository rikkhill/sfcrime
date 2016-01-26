# Multiclass Perceptron
import numpy as np
import pandas as pd
import helper.data as hd
import helper.constants as C
from algo.perceptron import Perceptron, Feature

data = hd.get_small_subsample()

# I can probably factor these into constants
labels = C.LABELS # 39 classes of crime
districts = pd.unique(data['PdDistrict'].ravel())
district_lookup = dict(zip(districts, range(1, len(districts) + 1)))

days = pd.unique(data['DayOfWeek'].ravel())
day_lookup = dict(zip(days, range(1, len(days) + 1)))

p = Perceptron(labels)

# I can factor these out to a Feature object in the perceptron module
def districtFeature(x):
	expression = np.zeros(len(districts), dtype = np.int)
	expression[district_lookup[x['PdDistrict']]] = 1
	return expression

def dayFeature(x):
	expression = np.zeros(len(days), dtype = np.int)
	expression[day_lookup[x['DayOfWeek']]] = 1
	return expression

print(dayFeature(data.ix[10,:]))