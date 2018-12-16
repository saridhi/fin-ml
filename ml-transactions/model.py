#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 01:02:10 2018

Logistic Regression model and some testing/metrics evaluated here

@author: dhirensarin
"""

import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score, precision_recall_curve


class Model:
    
    def __init__(self, features, labels, test_size=0.20):
        self.X = features
        self.y = labels
        self.test_size = test_size
        self.setup()
    
    #Perform oversampling to deal with imbalanced classes
    def setup(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size) 
        os = SMOTE(random_state=0)
        self.X_train, self.y_train = os.fit_sample(self.X_train, self.y_train)
        
    def train(self):
        #Tried KNN though Logistic performed better
        #self.classifier = KNeighborsClassifier(n_neighbors=5)  
        self.classifier = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000, penalty='l2')
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier

    def test(self, user_features = None):
        if (user_features is None):
            test_set = self.X_test
        else:
            test_set = user_features
        y_pred_prob = self.classifier.predict_proba(test_set)  
        y_pred = self.classifier.predict(test_set)  
        return y_pred_prob[:,1], y_pred
        #return self.metrics(y_pred)
    
    def metrics(self, y_pred, y_score):
        ret_dict = {'con_mat': confusion_matrix(self.y_test, y_pred),
                    'recall_score': recall_score(self.y_test, y_pred, average='binary'),
                    'accuracy_score': accuracy_score(self.y_test, y_pred, normalize=True),
                    'precision_score': precision_score(self.y_test, y_pred, average='binary'),
                    'f1_score': f1_score(self.y_test, y_pred, average='binary'),
                    'roc_auc_score': roc_auc_score(self.y_test, y_score)}
        return ret_dict
    
    def plot_pr_curve(self, y_pred, y_score):
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_score)
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, here='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        average_precision = average_precision_score(self.y_test, y_score)
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.show()
        return plt

