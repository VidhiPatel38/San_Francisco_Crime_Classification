import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def llfun(act, pred):
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

#z = zipfile.ZipFile('/Users/nikunjpatel/Desktop/DataMiningGroup/newData.zip')
train = pd.read_csv(open('train.csv'), parse_dates=['Dates'])[['DayOfWeek', 'PdDistrict', 'Category']]


# Separate test and train set out of orignal train set.
msk = np.random.rand(len(train)) < 0.8
knn_train = train[msk]
knn_test = train[~msk]
n = len(knn_test)

print("Original size: %s" % len(train))
print("Train set: %s" % len(knn_train))
print("Test set: %s" % len(knn_test))

# Prepare data sets
x = knn_train[['DayOfWeek', 'PdDistrict']]
y = knn_train['Category'].astype('category')
actual = knn_test['Category'].astype('category')
print("Here after split")


# Submit for K=3
test = pd.read_csv(open('test.csv'), parse_dates=['Dates'])
x_test = test[['DayOfWeek', 'PdDistrict']]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
outcomes = knn.predict(x_test)

submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in y.cat.categories:
    submit[category] = np.where(outcomes == category, 1, 0)
    print("Printing")
submit.to_csv('k_nearest_neigbour.csv', index = False)