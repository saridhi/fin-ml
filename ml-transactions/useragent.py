#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 00:51:43 2018

@author: dhirensarin
"""

import pickle
from model import Model

class UserAgent:
    
    def __init__(self, user_id):
        self.user_id = user_id
        
        
    def patrol():
        feature_df_enc = pickle.load(open("features.pck","rb"))
        model_features = feature_df_enc.loc[feature_df_enc.index != uuid.UUID('000e88bb-d302-4fdc-b757-2b1a2c33e7d6')]
        labels = user_df['is_fraudster'] * 1
        labels = labels[model_features.index]
        model = Model(feature_df_enc, labels)

        
     
if __name__ == "__main__":
    ua = UserAgent(uuid.UUID('0180632d-7737-42af-aaf0-95c2714d7854'))
    ua.patrol()