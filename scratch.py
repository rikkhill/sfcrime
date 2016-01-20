# Rikk's exploratory gubbins

import pandas as pd
import helper.data as hd
import helper.plot as plot

data = hd.get_n_training(100000)

# Remove borked lat/long
data = data[data.Y < 50]

plot.eventmap(data, 'Category')