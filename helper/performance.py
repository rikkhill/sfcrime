import numpy as np
import pandas as pd
from .constants import *
from .data import *

# Code vector
# Takes a data set and returns a vector of crime codes
def crime_codes(test_set):
    return test_set['Category'].apply(lambda x: CODE[x])

# Out-of-sample error
def oos_error(prediction, test_set):
    true_vals = crime_codes(test_set)
    indices = (prediction == true_vals).astype(int)
    proportion = indices.sum() / len(test_set)

    return proportion

# multiclass log loss, with a correction to put it in line with Kaggle
def mc_loss(pred, test_set, eps=1e-15, correction=40):
    # Map test set to binary matrix
    act = vec_to_binmatrix(crime_codes(test_set))

    # Clip predictions to (0, 1)
    pred = np.maximum(eps, pred)
    pred = np.minimum(1 - eps, pred)

    # Coerce dataframes into matrix
    if(type(pred) is pd.DataFrame):
        pred = pred.as_matrix()

    assert act.shape == pred.shape, "prediction and test must be the same dims"
    
    scores = []
    for i in range(0, len(pred)):
        #ll = sum(act[i]*np.log(pred[i]) + np.subtract(1,act[i])*np.log(np.subtract(1,pred[i])))
        ll = np.dot(act[i], np.log(pred[i]))
        ll = ll * -1.0/len(act[i])
        scores.append(ll)

    return (sum(scores) * correction) / len(scores)







# Precision, recall, F1
def scores(prediction, test_set):
    true_vals = crime_codes(test_set)
    df = pd.DataFrame(columns=['true', 'predict'])
    df['true'] = true_vals
    df['predict'] = prediction

    results = pd.DataFrame(columns=['Category', 'Precision', 'Recall', 'F1'], index=range(1, len(LABELS)+1))

    eps = 1e-12

    for i in range(1, len(LABELS) +1):
        results.loc[i, ['Category']] = LABELS[i-1]
        true_subset = df[df['true'] == i]
        predicted_subset = df[df['predict'] == i]
        correctly_predicted = df[(df['true'] == i) & (df['predict'] == i)]

        precision = (len(correctly_predicted) + eps) / (len(predicted_subset) + eps)

        recall = (len(correctly_predicted) + eps) / (len(true_subset) + eps)

        f1 = 2 * (precision * recall) / (precision + recall)

        results.loc[i, ['Precision']] = round(precision, 4)
        results.loc[i, ['Recall']] = round(recall, 4)
        results.loc[i, ['F1']] = round(f1, 4)


    return results






