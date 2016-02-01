# Rikk's exploratory gubbins
import pandas as pd
import helper.data as hd
import helper.plot as plot
import helper.constants as C

df = hd.get_training_data()

total = len(df)

# Filter for the top ~80%% of crimes
df = df[df.Category.isin([
                            #'LARCENY/THEFT',    # 19.92%
                            #'OTHER OFFENSES',   # 14.37%
                            #'NON-CRIMINAL',     # 10.51%
                            #'ASSAULT',          # 8.76%
                            'DRUG/NARCOTIC',    # 6.15%
                            #'VEHICLE THEFT',    # 6.13%
                            #'VANDALISM',        # 5.09
                            #'WARRANTS',         # 4.81%
                            #'BURGLARY'          # 4.19%
                            ])]

print(len(df)/total)

plot.eventmap(df, 'Category')