# Predict the relative frequency of crime types
import pandas as pd
import numpy as np
import helper.data as hd
import helper.performance as perf
import helper.plot as plot
import helper.constants as C

# Get a training split
#(train, test) = hd.get_training_split()
train = hd.get_training_data()
test = train

probs = []
count = len(train)

# Go through each label and find the probability

for i in C.LABELS:
	probs.append( len(train[ train['Category'] == i ]) / count )


fr = pd.DataFrame(index=range(C.TEST_ROWS))

fr[0] = range(C.TEST_ROWS)

for i, val in enumerate(probs):
	fr[i + 1] = val

#hd.prepare_submission(fr, "well_distributed")

print(perf.mc_loss(fr[0:len(test)], test))