# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

# 1. loads the data file;
wine_quality = pd.read_csv('C:/Users/karim/Documents/Imperial/Machine Learning/ProblemSets/Assignment1/winequality-red.csv',
                          sep = ';')
# 2. construct a new binary column “good wine” that indicates whether the wine is good 
# (which we define as having a quality of 6 or higher) or not;
def decide_good_wine(x):
    if x >= 6:
        return 1
    else:
        return 0
# good_wine = 1: it is good wine (quality >= 6)
wine_quality['good_wine'] = wine_quality.quality.apply(decide_good_wine)

# 3. splits the data set into a training data set (~50%) and a test data set (~50%) 
# — make sure you shuffle the record before the split;
# and,
# 4. normalises the data according to the Z-score transform;
wine_quality_mean = np.mean(wine_quality.iloc[:, 0:11], axis = 0)
wine_quality_std = np.std(wine_quality.iloc[:, 0:11], axis = 0)
wine_quality_normal = pd.DataFrame()
for i in list(range(11)):
    ap = ((wine_quality.iloc[:, i] - np.array(np.zeros(len(wine_quality.axes[0])) + wine_quality_mean[i])) / np.array(np.zeros(len(wine_quality.axes[0])) + wine_quality_std[i]))
    wine_quality_normal = pd.concat([wine_quality_normal, ap], axis = 1)
wine_quality_normal = pd.concat([wine_quality_normal, wine_quality.iloc[:, 11:13]], axis = 1)    

random_seed = 100
wine_quality_normal_shuffled = shuffle(wine_quality_normal, random_state = random_seed)
training_set_normal = wine_quality_normal_shuffled[0:round(len(wine_quality_normal_shuffled.axes[0])/2)]
test_set_normal = wine_quality_normal_shuffled[round(len(wine_quality_normal_shuffled.axes[0])/2):len(wine_quality_normal_shuffled.axes[0])]

# 5. loads and trains the k-Nearest Neighbours classifiers for k = 1, 6, 11, 16, ..., 500;
KNN_wine_classifier = []
for i in list(range(1, 500, 5)):
    KNN_wine_classifier.append(KNeighborsClassifier(n_neighbors = i))
    
# for i in list(range(len(KNN_wine_classifier))):
#     KNN_wine_classifier[i].fit(training_set_normal.iloc[:, 0:11], training_set_normal.iloc[:, 12])

# 6. evaluates each classifier using 5-fold cross validation and selects the best classifier;
# 100 items, each item contains a list of (five) cross-validation scores
scores = list()
counter=[]
for i in list(range(len(KNN_wine_classifier))):
    count=i
    counter.append(count)
    scores.append(list(cross_val_score(KNN_wine_classifier[i], training_set_normal.iloc[:, 0:11], training_set_normal.iloc[:, 12], cv = 5)))

# select the classifier with the higherest average score on the five folds
avg_scores = [sum(scores[i])/5 for i in list(range(len(scores)))]
max_score = max(avg_scores)
max_score_pos = np.argmax(avg_scores)
print('The best performed classifier (with random seed set as ' + str(random_seed) + ') is the ' + str(max_score_pos + 1) + 'th' + ' classifier, with an average validation score of ' + str(max_score) + '.')

plt.plot(counter, avg_scores)
plt.xlabel("No of k-tests undertaken from 1 to 500 with an interval of 5")
plt.ylabel("accuracy")


# 7. predicts the generalisation error using the test data set, as well as outputs the result in a confusion matrix.
# train this classifier again using all data (here the training_set_normal)
KNN_wine_classifier[max_score_pos].fit(training_set_normal.iloc[:, 0:11], training_set_normal.iloc[:, 12])

# prediction
test_score = KNN_wine_classifier[max_score_pos].predict(test_set_normal.iloc[:, 0:11])
print('Classification Report')

print(classification_report(y_true = test_set_normal.iloc[:, 12],
                            y_pred = test_score,
                            target_names = ['Not good wine', 'Good wine']))

x = confusion_matrix(y_true = test_set_normal.iloc[:, 12],
                     y_pred = test_score)
print('Confusion Matrix')
print(x)
print('Total Error Rate = ' + str((x[0, 1] + x[1, 0]) / sum(sum(x))))
print('Accuracy = ' + str(1 - (x[0, 1] + x[1, 0]) / sum(sum(x))))
print('Sensitivity = ' + str(x[0, 0] / (x[0, 0] + x[0, 1])))
print('Specificity = ' + str(x[1, 0] / (x[1, 0] + x[1, 1])))



#Additional content - KA


test_data=test_set_normal.iloc[:, 0:11]
test_y=test_set_normal.iloc[:, 12]

cfold_accuracy = []
ccounter_fold=[]
count=0    

for i in range(1,500,5):
    count = i
    ccounter_fold.append(count)
    knn_final=KNeighborsClassifier(n_neighbors=i)
    knn_final.fit(test_data, test_y)
    k_predict = knn_final.predict(test_data)
    cfold_accuracy.append(metrics.accuracy_score(test_y, k_predict))


plt.figure(1)
plot1, = plt.plot(ccounter_fold, cfold_accuracy)
plot2, = plt.plot(ccounter_fold, avg_scores)
plt.legend([plot2, plot1], ['Cross Validation - Training set', 'Accuracy on validation set'])
plt.xlabel("k-value")
plt.ylabel("accuracy")
