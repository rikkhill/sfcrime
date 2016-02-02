# Rikk tries logistic regression against various factors
import pandas as pd
import helper.data as hd
import helper.plot as plot
import helper.constants as C
import helper.performance as perf
import numpy as np
import statsmodels.api as sm
import os
import warnings
import sys
from datetime import datetime


# NAME YOUR MODEL!
modelname = "time_logit_regression"
path = "./models/%s" % modelname

print("Running model %s" % modelname)



# regressors should include numeric parameters you want in the regression
# dummies should include categorical parameters you want dummy variables for

regressors = ['Year']
dummies = ['DayOfWeek', 'Hour', 'Month'] # , 'Year']


# Transform dataframe into how we want it
def munge(data):
    # Lets give ourselves an hour, month and year column
    times = pd.to_datetime(data.Dates, infer_datetime_format=True)

    hour = times.apply(lambda x: x.hour)
    hour.name = 'Hour'

    month = times.apply(lambda x: x.month)
    month.name = 'Month'

    year = times.apply(lambda x: x.year)
    year.name = 'Year'

    data = pd.concat([data, hour, month, year], axis=1)

    return data


logit_models = []

# Check to see if model already exists
# Load it if it does
# Fit it if it doesn't
if os.path.isdir(path) and os.path.isfile("%s/built.txt" % path):
    # Model has already been built; load it
    print("Model %s already built. Loading..." % modelname)
    from statsmodels.iolib.smpickle import load_pickle
    for i in C.LABELS:
        print("\t Loading %s..." % i)
        filesafe_name = "".join([c for c in i if c.isalpha() or c.isdigit()]).rstrip()
        if os.path.isfile("%s/%s.pickle" % (path, filesafe_name)):
            logit = load_pickle("%s/%s.pickle" % (path, filesafe_name))
            logit_models.append(logit)
        else:
            print("\t\t%s model doesn't exist. Loading empty model" % i)
            logit_models.append(0)

else:
    # Model hasn't been built
    # Make the directory
    if not os.path.isdir(path):
        os.makedirs(path)

    # fit the logit models
    print("\nModel %s does not exist. Fitting..." % modelname)
    print("\nLoading training data")
    (df, validation) = hd.get_training_split()

    # Get rid of North Pole crime
    df = df[df.Y < 50]
    df = munge(df)

    dummy_categories = pd.get_dummies(df['Category'], prefix='category')

    dummy_sets = []
    for d in dummies:
        dummy_sets.append(pd.get_dummies(df[d], prefix=d))

    print("\nTraining models...")

    for i in C.LABELS:
        print("\tTraining %s (%d/%d)" % (i, C.CODE[i], len(C.LABELS)))
        response_s = "category_%s" % i
        rframe = pd.DataFrame(data=dummy_categories[response_s]).join(df[regressors])
        for d in dummy_sets:
            rframe = rframe.join(d)
        rframe['intercept'] = 1.0
        train_cols = rframe.columns[1:]
        logit = sm.Logit(rframe[response_s], rframe[train_cols])
        try:
            # Ignore known warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = logit.fit()
                # Save to file
                filesafe_name = "".join([c for c in i if c.isalpha() or c.isdigit()]).rstrip()
                result.save("%s/%s.pickle" % (path, filesafe_name), remove_data=True)
                print("\t\tTrained logit model written to file %s" % filesafe_name)
        except np.linalg.linalg.LinAlgError as e:
            print(e)
            result = 0
        finally:
            logit_models.append(result)

    # Touch built.txt
    open("%s/built.txt" % path, 'a').close()


# Predict loss
(train, validation) = hd.get_training_split()

validation = munge(validation)

print("\nPredicting on cross-validation set...")
validation_preds = pd.DataFrame(index=range(len(validation)))
validation_preds['Id'] = pd.Series(range(len(validation))) 

test_coords = pd.DataFrame(data = validation[regressors])

for d in dummies:
    test_coords = test_coords.join(pd.get_dummies(validation[d], prefix=d))

test_coords.loc[:, 'intercept'] = 1

for i in C.LABELS:
    print("\t\tPredicting for %s" % i)
    model = logit_models[C.CODE[i] - 1]
    if model != 0:
        preds = model.predict(test_coords)
    else:
        preds = 0
    validation_preds[i] = preds

print("\tCalculating loss...")
print("\tMC Loss: %3.4f" % perf.mc_loss(validation_preds, validation))


print("\nPredicting on test set...")

test = hd.get_test_data()
test = munge(test)
final_preds = pd.DataFrame(index=range(len(test)))
final_preds['Id'] = pd.Series(range(len(test))) 

test_coords = pd.DataFrame(data = test[regressors])

for d in dummies:
    test_coords = test_coords.join(pd.get_dummies(test[d], prefix=d))

test_coords.loc[:, 'intercept'] = 1

for i in C.LABELS:
    print("\tPredicting for %s" % i)
    model = logit_models[C.CODE[i] - 1]
    if model != 0:
        preds = model.predict(test_coords)
    else:
        preds = 0
    final_preds[i] = preds

print("\nPreparing file for submission")
hd.prepare_submission(final_preds, modelname)
print("\tPrepared file %s.zip" % modelname )