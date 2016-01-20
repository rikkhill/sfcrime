import pandas as pd
import scipy.sparse as sp
import numpy as np
import zipfile
from .constants import LABELS 


try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED
    
def get_training_data():
	return pd.read_csv("./data/train.csv").fillna(value=0)

def get_n_training(n):
	return get_training_data().ix[0:n, :]

def get_training_subset():
	return pd.read_csv("./data/training_subset.csv").fillna(value=0)

def get_validation_subset():
	return pd.read_csv("./data/validation_subset.csv").fillna(value=0)

def get_small_subsample():
	return pd.read_csv("./data/small_subsample.csv").fillna(value=0)

def get_test_data():
	return pd.read_csv("./data/test.csv").fillna(value=0)

def get_n_test(n):
	return get_test_data().ix[0:n, :]

# Get an 80/20 training/validation split from the training data
def get_training_split():
	data = get_training_data()
	trainsize = int(len(data)//1.25)
	return( data[0:trainsize], data[trainsize:] )

def vec_to_binmatrix(vec, pad = 40):
	length = vec.shape[0]
	indptr = range(length + 1)
	data = np.ones(length)
	matrix = sp.csr_matrix((data, vec, indptr)).toarray()
	matrix[:, 0] = np.array(range(0, length))
	width = matrix.shape[1]
	padded = np.pad(matrix, ((0,0),(0, pad - width)), 'constant', constant_values=(0, 0))

	return padded

def submit(data, filename):
	print("Formatting data")
	matrix = vec_to_binmatrix(data)

	print("Writing to file")
	prepare_submission(matrix, filename)

def prepare_submission(data, filename):
	# Make sure the submission is the right shape
	assert data.shape == (884262, 40), "Submission has the wrong dimensions"

	# Write to file
	f = open("./output/%s.csv" % filename, "w")
	header = ["Id"] + LABELS
	f.write(",".join(header) + "\n")
	f.close()

	# Stupid mixed file modes
	f = open("./output/%s.csv" % filename, "ab")
	np.savetxt(f, data, fmt = "%1.0d", newline = "\n", delimiter = ",")
	f.close()

	# Zip it up
	zippy = zipfile.ZipFile("./output/%s.zip" % filename, mode = 'w')
	try:
		zippy.write("./output/%s.csv" % filename, compress_type=compression)
	finally:
		zippy.close()

	