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

# 1. loads the data file;
wine_quality = pd.read_csv('/Users/Veronique/Desktop/ICBA/Nov_Dec Term/Machine Learning/Assignment/Assignment1/GroupWork_Data/winequality-red.csv',
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
random_seed = 100
wine_quality_shuffled = shuffle(wine_quality, random_state = random_seed)
training_set = wine_quality_shuffled[0:round(len(wine_quality_shuffled.axes[0])/2)]
test_set = wine_quality_shuffled[round(len(wine_quality_shuffled.axes[0])/2):len(wine_quality_shuffled.axes[0])]

# 4. normalises the data according to the Z-score transform;
wine_mean_training = np.mean(training_set.iloc[:, 0:11], axis=0)
wine_std_training = np.std(training_set.iloc[:, 0:11], axis=0)
training_set_normal = pd.DataFrame()
for i in list(range(11)):
    ap = ((training_set.iloc[:, i] - np.array(np.zeros(len(training_set.axes[0])) + wine_mean_training[i])) / np.array(np.zeros(len(training_set.axes[0])) + wine_std_training[i]))
    training_set_normal = pd.concat([training_set_normal, ap], axis = 1)
training_set_normal = pd.concat([training_set_normal, training_set.iloc[:, 11:13]], axis = 1)

# 5. loads and trains the k-Nearest Neighbours classifiers for k = 1, 6, 11, 16, ..., 500;
KNN_wine_classifier = []
for i in list(range(1, 500, 5)):
    KNN_wine_classifier.append(KNeighborsClassifier(n_neighbors = i))
    
# for i in list(range(len(KNN_wine_classifier))):
#     KNN_wine_classifier[i].fit(training_set_normal.iloc[:, 0:11], training_set_normal.iloc[:, 12])

# 6. evaluates each classifier using 5-fold cross validation and selects the best classifier;
# 100 items, each item contains a list of (five) cross-validation scores
scores = list()
for i in list(range(len(KNN_wine_classifier))):
    scores.append(list(cross_val_score(KNN_wine_classifier[i], training_set_normal.iloc[:, 0:11], training_set_normal.iloc[:, 12], cv = 5)))
# select the classifier with the higherest average score on the five folds
avg_scores = [sum(scores[i])/5 for i in list(range(len(scores)))]
max_score = max(avg_scores)
max_score_pos = np.argmax(avg_scores)
print('The best performed classifier (with random seed set as ' + str(random_seed) + ') is the ' + str(max_score_pos + 1) + 'th' + ' classifier, with an average validation score of ' + str(max_score) + '.')

# 7. predicts the generalisation error using the test data set, as well as outputs the result in a confusion matrix.
# train this classifier again using all data (here the training_set_normal)
KNN_wine_classifier[max_score_pos].fit(training_set_normal.iloc[:, 0:11], training_set_normal.iloc[:, 12])

# normalise test_set (seperate from training_set)
wine_mean_test = np.mean(test_set.iloc[:, 0:11], axis=0)
wine_std_test = np.std(test_set.iloc[:, 0:11], axis=0)
test_set_normal = pd.DataFrame()
for i in list(range(11)):
    ap = ((test_set.iloc[:, i] - np.array(np.zeros(len(test_set.axes[0])) + wine_mean_test[i])) / np.array(np.zeros(len(test_set.axes[0])) + wine_std_test[i]))
    test_set_normal = pd.concat([test_set_normal, ap], axis = 1)
test_set_normal = pd.concat([test_set_normal, test_set.iloc[:, 11:13]], axis = 1)

# prediction
test_score = KNN_wine_classifier[max_score_pos].predict(test_set.iloc[:, 0:11])
print('Classification Report')
print(classification_report(y_true = test_set_normal.iloc[:, 12],
                            y_pred = test_score,
                            target_names = ['Not good wine', 'Good wine']))
print('Confusion Matrix')
print(confusion_matrix(y_true = test_set_normal.iloc[:, 12],
                       y_pred = test_score))



