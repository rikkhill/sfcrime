# Rikk's misguided attempt to see if we can find "crime centroids"
import pandas as pd
import helper.data as hd
import helper.plot as plot
import helper.constants as C
import numpy as np
import statsmodels.api as sm

df = hd.get_training_data()
"""
df = df[df.Y < 50] # Filter out the North Pole

# Get mean values for lat and long for each category of crime
centroids = {}

for i in C.LABELS:
    centroids[i] =  np.array([np.mean(df[df.Category == i]['X']),
                    np.mean(df[df.Category == i]['Y'])])

"""

# Lets try logistic regression by district

dummy_districts = pd.get_dummies(df['PdDistrict'], prefix='district')
print(dummy_districts.head())