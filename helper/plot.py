import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .constants import *


# Take the lat and long of this subset of data and plot them in 2D by colname
def eventmap(data, colname):
    
    # Get rid of all those San Franciscan crimes at the North Pole...
    data = data[data.Y < 50]

    vals = pd.unique(data[colname].ravel())
    assert len(vals) <= 56, "Plot can only support up to 56 categories"

    # Every distinct value has  
    valcode = dict(zip(vals, range(1, len(vals) + 1)))

    x = data['X'].tolist()
    y = data['Y'].tolist()
    colours = list(map(lambda x: COLOURS[valcode[x]], data[colname]))

    plt.scatter(x, y, s=10, c=colours, alpha=0.5)
    plt.show()
