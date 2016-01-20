# Written by Rikk
# Sets up various datasets

import helper.data as hd

print("Beginning setup...")

# Make training and validation subsets
print("Splitting training data")
(training, validate) = hd.get_training_split()

print("Writing training subset")
training.to_csv('./data/training_subset.csv')

print("Writing validation subset")
validate.to_csv('./data/validation_subset.csv')

# Make small 100 sample subset for easy in-file browsing
print("Writing small (100 sample) subset to small_subsample.csv")
training.to_csv('./data/small_subsample.csv')


print("Setup complete")