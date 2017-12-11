# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:22:16 2017

@author: Ryan
"""

import matplotlib.pyplot as plt
import csv
import re
import pandas as pd
import sklearn
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.learning_curve import learning_curve

#Data loading
messages = pd.read_csv('C:/Users/karim/Documents/Imperial/Machine Learning/ProblemSets/Assignment2/SMSSpamCollection.txt',sep='\t', header=None,
                           names=["label", "messages"])

#Data processing
def data_processing(p):
    remove_number_punc = re.sub("[^a-zA-Z]", " ", p)
    convert_to_lower_letter = remove_number_punc.lower()
    return convert_to_lower_letter

messages['messages'] = messages['messages'].apply(data_processing)
print(messages.head())
messages.groupby('label').describe()
messages['messages'].describe()



#Data shuffling and segmenting
random_seed = 100

clean_messages_shuffled = shuffle(messages, random_state = random_seed)

training_set = clean_messages_shuffled [0:round(len(clean_messages_shuffled.axes[0])/2.2288)]

validation_set =clean_messages_shuffled [round(len(clean_messages_shuffled.axes[0])/2.23):round(len(clean_messages_shuffled.axes[0])/1.592)]

test_set=clean_messages_shuffled[round(len(clean_messages_shuffled.axes[0])/1.5925):round(len(clean_messages_shuffled.axes[0])/1)]
len(test_set)

istrain_ham = list(test_set[test_set['label'] == 'ham']['label'])
len(istrain_ham)


print(len(training_set),len(validation_set),len(test_set))

#Q4
class NaiveBayesForSpam:
    def train (self, hamMessages, spamMessages):
        self.words = set (' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros (2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len (spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len (hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
        self.likelihoods = np.array (self.likelihoods).T
        
    def train2 (self, hamMessages, spamMessages):
        self.words = set (' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros (2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len (spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        spamkeywords = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len (hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            if prob1 * 20 < prob2:
                self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
                spamkeywords.append (w)
        self.words = spamkeywords
        self.likelihoods = np.array (self.likelihoods).T

    def predict (self, message):
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower():  # convert to lower-case
                posteriors *= self.likelihoods[:,i]
            else:                                   
                posteriors *= np.ones (2) - self.likelihoods[:,i]
            posteriors = posteriors / np.linalg.norm (posteriors)  # normalise
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]    

    def score (self, messages, labels):
        confusion = np.zeros(4).reshape (2,2)
        for m, l in zip (messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion
    
#Q6
train_ham = training_set.loc[training_set['label'] == 'ham']
train_spam = training_set[training_set['label'] == 'spam']

classifier_train1 = NaiveBayesForSpam()
classifier_train2 = NaiveBayesForSpam()

classifier_train1.train(train_ham['messages'].tolist(), train_spam['messages'].tolist())
classifier_train2.train2(train_ham['messages'].tolist(), train_spam['messages'].tolist())

#Q7
classifier_train1.score(validation_set['messages'],validation_set['label'])
classifier_train2.score(validation_set['messages'],validation_set['label'])

#Q10
classifier_train2.score(test_set['messages'],test_set['label'])







    



      

                 




