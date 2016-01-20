# All crimes are Larceny and Theft
import pandas as pd
import numpy as np
import helper.data as hd
import helper.plot as plot
import helper.constants as C

code = C.CODE['LARCENY/THEFT']

vec = np.zeros((C.TEST_ROWS,), dtype = np.int)
vec.fill(code)


hd.submit(vec, 'alltheft')