print("Importing necessary libraries...")
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

def llfun(act, pred):
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

#z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(open('train.csv'), parse_dates=['Dates'])[['X', 'Y', 'Category']]

print("Separate test and train set out of orignal train set...")
msk = np.random.rand(len(train)) < 0.8
rfc_train = train[msk]
rfc_test = train[~msk]
n = len(rfc_test)

print("Printing dataset lengths...")
print("Original size: %s" % len(train))
print("Train set: %s" % len(rfc_train))
print("Test set: %s" % len(rfc_test))

print("Preparing data sets...")
x = rfc_train[['X', 'Y']]
y = rfc_train['Category'].astype('category')
actual = rfc_test['Category'].astype('category')

print("Making predictions...")
test = pd.read_csv(open('test.csv'), parse_dates=['Dates'])
x_test = test[['X', 'Y']]
etc = ExtraTreesClassifier(n_estimators=25)
etc.fit(x, y)
outcomes = etc.predict(x_test)

print("Submitting the results...")
submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in y.cat.categories:
    submit[category] = np.where(outcomes == category, 1, 0)
    print("Printing")
    
submit.to_csv('extratree_criminals.csv', index = False)