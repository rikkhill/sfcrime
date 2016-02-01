# All crimes are Larceny and Theft
import pandas as pd
import numpy as np
import helper.data as hd
import helper.performance as perf
import helper.plot as plot
import helper.constants as C

# Get a training split
(train, test) = hd.get_training_split()

# Make a vector where every code is LARCENY/THEFT
code = C.CODE['LARCENY/THEFT']
vec = np.zeros((len(test),), dtype = np.int)
vec.fill(code)

# What's the out-of-sample error?
oos = perf.oos_error(vec, test)
print(oos)

sc = perf.scores(vec, test)
print(sc)

prediction = hd.vec_to_binmatrix(vec)

print(perf.mc_loss(prediction, test))

#hd.submit(vec, 'alltheft')