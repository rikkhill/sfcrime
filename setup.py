# Written by Rikk
# Sets up various datasets

import helper.data as hd

print("Beginning setup...")

# Make training and validation subsets
print("Splitting training data")
(training, validate) = hd.get_training_split()

print("Writing training subset")
training.to_csv('./data/training_subset.csv', index=False)

print("Writing validation subset")
validate.to_csv('./data/validation_subset.csv', index=False)

# Make small 100 sample subset for easy in-file browsing
print("Writing small (100 sample) subset to small_subsample.csv")
training[0:100].to_csv('./data/small_subsample.csv', index=False)


print("Setup complete")