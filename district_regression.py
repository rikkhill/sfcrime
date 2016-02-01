# Rikk's misguided attempt to determine crime by location
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


# NAME YOUR MODEL!
modelname = "geoposition_logit"
path = "./models/%s" % modelname

print("Running model %s" % modelname)

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
        if os.isfile("%s/%s.pickle" % (path, filesafe_name)):
            logit = load_pickle("%s/%s.pickle" % (path, filesafe_name))
            logit_models.append(logit)
        else:
            print("\t\t%s model doesn't exist. Loading empty model")
            logit_models.append(0)

else:
    # Model hasn't been built
    # Make the directory
    if not os.path.isdir(path):
        os.makedirs(path)

    # fit the logit models
    print("Model %s does not exist. Fitting..." % modelname)
    print("\nLoading training data")
    (df, validation) = hd.get_training_split()

    # Get rid of North Pole crime
    df = df[df.Y < 50] 

    dummy_districts = pd.get_dummies(df['PdDistrict'], prefix='district')
    dummy_categories = pd.get_dummies(df['Category'], prefix='category')

    print("\nTraining models...")

    for i in C.LABELS:
        print("\tTraining %s (%d/%d)" % (i, C.CODE[i], len(C.LABELS)))
        response_s = "category_%s" % i
        rframe = pd.DataFrame(data=dummy_categories[response_s]).join(df[['X', 'Y']])
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
                result.save("%s/%s.pickle" % (path, filesafe_name)
                print("\tTrained logit model written to file %s" % filesafe_name)
        except np.linalg.linalg.LinAlgError as e:
            print(e)
            result = 0
        finally:
            logit_models.append(result)

    # Touch built.txt
    open("%s/built.txt" % path, 'a').close()




print("\nPredicting on cross-validation set...")
validation_preds = pd.DataFrame(index=range(len(test)))
validation_preds['Id'] = pd.Series(range(len(test))) 

test_coords = pd.DataFrame(data = test[['X', 'Y']])
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
final_preds = pd.DataFrame(index=range(len(test)))
final_preds['Id'] = pd.Series(range(len(test))) 

test_coords = pd.DataFrame(data = test[['X', 'Y']])
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
hd.prepare_submission(all_preds, modelname)
print("\tPrepared file %s.csv.zip" % modelname )