import numpy as np
import pandas as pd
from .constants import *

# Code vector
# Takes a data set and returns a vector of crime codes
def crime_codes(test_set):
	return test_set['Category'].apply(lambda x: CODE[x])

# Out-of-sample error
def oos_error(prediction, test_set):
	true_vals = crime_codes(test_set)
	indices = (prediction == true_vals).astype(int)
	proportion = indices.sum() / len(test_set)

	return proportion
