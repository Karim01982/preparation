# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:10:13 2017

@author: karim
"""
import pandas as pd
import scipy as sp
import numpy as np
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#1. Importing the data
data_set = pd.read_csv("C:/Users/karim/Documents/Imperial/Machine Learning/ProblemSets/Assignment1/winequality-red.csv", sep=";")

#2.New binary column
data_set.iloc[:,[11]]
data_set['ww_quality'] = data_set.quality.apply(lambda x: "good_wine" if x>=6 else "bad_wine")
data_set.head(10)

#3. & 4. Normalise the data using the Z-score transform and Split the data 50:50 between training and test data

sorted_data = shuffle(data_set)
sorted_data.head(10)
quality_column=sorted_data.iloc[:,12]
data_excqual = sorted_data.iloc[:,0:11]
z_score_transform = data_excqual.apply(zscore)
z_score_transform['ww_quality']=quality_column

train, test = sklearn.model_selection.train_test_split(z_score_transform, test_size=0.5)

#5.Load and train K-nearest neighbours
x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
y_column=["ww_quality"]

accuracy = []
counter=[]
count=0
for i in range(1,500,5):
    count = i
    counter.append(count)
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train[x_columns], train[y_column])
    k_predict = knn.predict(test[x_columns])
    accuracy.append(metrics.accuracy_score(test[y_column], k_predict))

labels=['kvalue','accuracy']
k_output =pd.DataFrame(list(zip(counter, accuracy)), columns = labels)
print(k_output)

#6a. 5-fold cross validation - on entire dataset

total_y = z_score_transform.iloc[:,11]
total_data = z_score_transform.iloc[:,0:11]

fold_accuracy = []
counter_fold=[]
count=0    

for i in range(1,500,5):
    count = i
    counter_fold.append(count)
    knn=KNeighborsClassifier(n_neighbors=i)
    k_score = cross_val_score(knn, total_data, total_y, cv=5, scoring = 'accuracy')
    fold_accuracy.append(k_score.mean())

kfold_output = pd.DataFrame(list(zip(counter_fold, fold_accuracy)), columns = labels)
#Optimal K-range between k=76 and k=121

##Side-test: comparing classification / confusion matrix with 7
knn_total=KNeighborsClassifier(n_neighbors=71)
knn_total.fit(total_data, total_y)
total_predict=knn_total.predict(total_data)

print(classification_report(total_y, total_predict, labels=None))
print(confusion_matrix(total_y, total_predict, labels=None))



#6b. 5-fold cross validation - on training dataset
training_data=train.iloc[:,0:11]
training_y=train.iloc[:,11]

bfold_accuracy = []
bcounter_fold=[]
count=0    

for i in range(1,500,5):
    count = i
    bcounter_fold.append(count)
    knn=KNeighborsClassifier(n_neighbors=i)
    k_score = cross_val_score(knn, training_data, training_y, cv=5, scoring = 'accuracy')
    bfold_accuracy.append(k_score.mean())

kfold_outputb = pd.DataFrame(list(zip(bcounter_fold, bfold_accuracy)), columns = labels)
#Optimal K-range between k=101 and k=126

##Side-test: comparing classification / confusion matrix with 7
knn_training=KNeighborsClassifier(n_neighbors=116)
knn_training.fit(training_data, training_y)
training_predict=knn_total.predict(training_data)

print(classification_report(training_y, training_predict, labels=None))
print(confusion_matrix(training_y, training_predict, labels=None))

#7 Predict generalisation error using the test set and outputs confusion matrix

test_data=test.iloc[:,0:11]
test_y=test.iloc[:,11]

cfold_accuracy = []
ccounter_fold=[]
count=0    

for i in range(1,500,5):
    count = i
    ccounter_fold.append(count)
    knn=KNeighborsClassifier(n_neighbors=i)
    k_score = cross_val_score(knn, test_data, test_y, cv=5, scoring = 'accuracy')
    cfold_accuracy.append(k_score.mean())

kfold_outputc = pd.DataFrame(list(zip(ccounter_fold, cfold_accuracy)), columns = labels)
#Optimal k=71

knn_final=KNeighborsClassifier(n_neighbors=71)
knn_final.fit(test_data, test_y)
final_predict=knn_final.predict(test_data)

print(classification_report(test_y, final_predict, labels=None))
print(confusion_matrix(test_y, final_predict, labels=None))

