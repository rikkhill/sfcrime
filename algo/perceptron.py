# Multiclass Perceptron

import numpy as np


class Perceptron:
	def __init__(self, labels=[]):
		self.labels = labels
		self.feature_set = []
		
		self.weights = np.array([])

	# Run for n iterations
	def iterate(self, n):
		for i in range(n):
			print(i)	


	def _build_weights(self):
		# Todo
		return False


class Feature:
	def __init__(self, fn, size):
		self.size = size
		self.fn = fn

	def express(self, sample):
		return False