import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
import numpy as np
 
#Load Data with pandas, and parse the first column into datetime
train = pd.read_csv("/Users/nikunjpatel/Desktop/DataMiningGroup/train.csv",parse_dates = ['Dates'])
test = pd.read_csv("/Users/nikunjpatel/Desktop/DataMiningGroup/test.csv", parse_dates = ['Dates'])
#Convert crime labels to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)
 
#Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour) 
address = pd.get_dummies(train.Address)         
x = pd.get_dummies(train.X)
y = pd.get_dummies(train.Y)
#Build new array
train_data = pd.concat([hour, days, district], axis=1)

train_data['crime']=crime
 
#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
 
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)  
test_data = pd.concat([hour, days, district], axis=1)
 
training, validation = train_test_split(train_data, train_size=.60)
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
 
features2 = [x for x in range(0,24)]
features = features + features2

  
#Logistic Regression for comparison'''
model = LogisticRegression(C=.01)
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])
#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('LogisticRegression.csv', index = True, index_label = 'Id' )

#Naive Bayes for Prediction
model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])
#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('NaiveBayes1.csv', index = True, index_label = 'Id' )

