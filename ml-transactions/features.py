#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 19:40:59 2018

Class to create features from the database user and transaction tables. 
This saves relevant features as a pickle serialized file for later consumption.

@author: dhirensarin
"""

import csv
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.orm import sessionmaker
from table_obj import Users, Transactions
import numpy as np
from settings import conn_str
from helper import orm_to_df
import datetime
from datetime import timedelta
from collections import Counter
import pickle
    
class Features:
    
    def __init__(self, session):
        self.session = session
        self.user_df = None
        self.trans_df = None
        
    def get_users(self):
        s = self.session()
        users = s.query(Users).all()
        
        #Normalise the country code
        country_dict = {}
        country_reader = csv.reader(open('countries.csv', 'r'))
        next(country_reader)  # Skip the header row.
        for line in country_reader:
            country_dict[line[0].upper()]=line[2].upper()

        self.user_df = orm_to_df(users)
        countries = [country_dict[i] if i in country_dict.keys() else i for i in self.user_df['country']]
        self.user_df['country'] = countries
        self.user_df.index = self.user_df['id']
        self.user_df.sort_index(inplace=True)
        return self.user_df
    
    def get_transactions(self):
        s = self.session()
        print('Retrieving Transactions...')
        transactions = s.query(Transactions).all()
        print('Done...adjusting transaction amounts...')
        self.trans_df = orm_to_df(transactions)
        self.trans_df.index = self.trans_df['id']
        self.trans_df.sort_index(inplace=True)
        self.trans_df = self.tx_replace_amount()
        print('Done!')
        return self.trans_df
    
    #Function to replace all amounts into USD for direct comparison
    def tx_replace_amount(self):
        s = self.session()
        query = "Select tx.id, Case When tx.currency = 'USD'  then tx.amount/POWER(10, cd.exponent) Else (tx.amount/Power(10,cd.exponent))*fx.rate End as dollar_amount FROM transactions tx INNER JOIN currency_details as cd on tx.currency = cd.ccy LEFT JOIN fx_rates AS fx on fx.ts = (SELECT MAX(ts) FROM fx_rates WHERE ts <=  tx.created_date) AND fx.ccy = tx.currency AND tx.currency != 'USD' AND fx.base_ccy='USD' ORDER BY tx.id;;"
        results = list(s.execute(query))
        self.trans_df.sort_index(inplace=True)
        self.trans_df['amount_usd'] = [i[1] for i in results]
        
        return self.trans_df
        
    
    def one_hot_replace(self, enc):
        return pd.get_dummies(enc)
    
    #One hot encode certain categorical fields
    def encode(self, feature_df):
        onehot_encoded_kyc = pd.DataFrame(self.one_hot_replace(feature_df['kyc']))
        feature_df = feature_df.drop(columns='kyc')
        
        onehot_encoded_age = pd.DataFrame(self.one_hot_replace(feature_df['birth_year']))
        feature_df = feature_df.drop(columns='birth_year')
        v_stack = pd.concat([feature_df, onehot_encoded_kyc, onehot_encoded_age], axis=1)
        return v_stack
        
    def create(self):
        userid_group = self.trans_df.groupby(['user_id'])
        
        #Number of transactions
        n_trans = userid_group.size()
        ind = n_trans.index
        
        #Transaction country count where it equals the User country
        trans_country = self.trans_df.groupby(['user_id', 'merchant_country']).size().reset_index('merchant_country')
        all_countries = pd.DataFrame(self.user_df['country']).join(trans_country)
        country_count = all_countries.groupby(all_countries.index).apply(lambda x:Counter(x['country']==x['merchant_country'])[True])
        country_count = country_count[ind]
    
        #Boolean for checking total number of transactions beneath a threshold (1000 in this case)
        trans_min = (n_trans < 1000).astype('int')
        kyc = self.user_df.loc[ind, 'kyc'] # to one hot encode
        
        #Calculate age
        now_year = datetime.datetime.now().year
        age = now_year - self.user_df.loc[ind, 'birth_year']
        
        #Bin ages to analysed ranges
        age = pd.cut(age, [16, 24, 32, 38, 100]) # to one hot encode
        #age_susp = age.apply(lambda x: 1 if int(x) in range(16,25))
        
        #Capture Failed sign ins grouped by <=2 and >=2
        failed_signin = self.user_df.loc[ind, 'failed_sign_in_attempts']
        failed_signin[failed_signin.isin([1,2])] = 1
        failed_signin[failed_signin>2] = 0
        
        #Number of transaction types per Card Payment and Topup as feature
        cp_sizes = self.trans_df[self.trans_df['type']=='CARD_PAYMENT'].groupby(['user_id']).size()
        tu_sizes = self.trans_df[self.trans_df['type']=='TOPUP'].groupby(['user_id']).size()
        
        cp_percent = cp_sizes.divide(n_trans).fillna(0)
        tu_percent = tu_sizes.divide(n_trans).fillna(0)
        
        #Transactions within a minute of each other
        trans_df_copy = self.trans_df.copy()
        trans_times = trans_df_copy.sort_values(['user_id', 'created_date']).groupby('user_id')['created_date']
        diff_trans = lambda x: Counter(x.diff() < timedelta(minutes=1))[True]
        trans_time_count = trans_times.apply(diff_trans)
        num_countries = userid_group['merchant_country'].nunique()
        
        #Repeat transactions in the same day
        trans_df_copy = self.trans_df.copy()
        trans_df_copy['created_date'] = [i.date() for i in trans_df_copy['created_date']]
        repeat_trans = lambda x: sum([i for i in Counter(str(x['created_date'])+str(x['amount'])).values() if i!=1])
        trans_repeats = trans_df_copy.groupby(['user_id']).apply(repeat_trans)
        
        #Other stats around the amounts
        max_amount = userid_group['amount_usd'].max()
        min_amount = userid_group['amount_usd'].min()
        range_amount = max_amount.subtract(min_amount)
        std_amount = userid_group['amount_usd'].std()/np.sqrt(n_trans)
        
        #Capture outliers
        def outliers(x):
            median = np.median(x)
            diff = np.sum((x - median)**2)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)
            return med_abs_deviation
        
        outlier_amount = userid_group['amount_usd'].apply(lambda x: outliers(x))
        
        v_stack = pd.concat([kyc, age, cp_percent, tu_percent, trans_time_count, num_countries, trans_repeats, range_amount, std_amount, trans_min, country_count, outlier_amount, failed_signin], axis=1)
        
        v_stack = self.encode(v_stack)
        v_stack = v_stack.fillna(0)
        
        return v_stack
        
if __name__ == "__main__":
    engine = create_engine(conn_str, echo=False)
    session = sessionmaker()
    session.configure(bind=engine)
    
    from features import Features
    features = Features(session)
    
    user_df = features.get_users()
    trans_df = features.get_transactions()
    
    print('Creating Features...')
    #Create features and store them
    feature_df = features.create()
    labels = user_df['is_fraudster'] * 1
    labels = labels[feature_df.index]
    
    print('Saving Features...')
    all_data = pd.concat([feature_df, labels], axis=1)
    pickle.dump(all_data,  open('features.pck','wb') )
    print('Done.')

