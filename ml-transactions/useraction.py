#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 00:51:43 2018

@author: dhirensarin
"""

import pickle
from model import Model
from sklearn.metrics import average_precision_score, precision_recall_curve
import uuid

class UserAction:
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.precision_threshold = 0.20 #Precision Threshold 
        self.prob_threshold = 0.70 #Probability threshold
        
    def patrol(self):
        #Retrieve data from features pickle saved when running features.py
        model_features = pickle.load(open("features.pck","rb"))
        
        #Extract the labels/y-value from the last column
        feature_rows = model_features.iloc[:,0:model_features.shape[1]-1]
        labels = model_features.iloc[:,len(model_features.columns)-1]
      
        user_features = feature_rows.loc[self.user_id, :]
        user_features = user_features.to_frame().transpose()
        #Train and test model to be used
        model = Model(feature_rows, labels, test_size=0.20)
        classifier = model.train()
        prob, pred = model.test()   
        
        #Some evaluation metrics printed out
        print(model.metrics(pred, prob))    
        
        #Retrieve precision, recall and thresholds to determine user signaling
        precision, recall, thresholds = precision_recall_curve(model.y_test, prob)
        
        prob, pred = model.test(user_features)   
        
        signal = "NO_FRAUD"
        
        if (pred == 1):
            
            #Alert agent at minimum
            signal = 'ALERT'
            
            ind  = len(thresholds[thresholds<prob])
            
            #Get precision and recall for the corresponding probability threshold of outcome
            precision = precision[ind]
            recall = recall[ind]
            
            if precision < self.precision_threshold and prob > self.prob_threshold:
                signal = 'BOTH'
            elif precision > self.precision_threshold:
                signal = 'LOCK'
            
        return signal

        
     
if __name__ == "__main__":
    ua = UserAction(uuid.UUID('fec955c7-cba4-42e5-9f04-b54e6072612d'))
    print(ua.patrol())
