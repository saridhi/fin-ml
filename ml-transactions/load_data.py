#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:11:27 2018
Script to load data into sql database
@author: dhirensarin
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import csv
from table_obj import CurrencyDetails, FxRates, Users, Transactions
from settings import conn_str

def load_currency_details(session, filename='currency_details.csv'):
    s = session()
    print('Loading currency details...')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row.
        for row in reader:
            row[3] = False if row[3]=='FALSE' else True
            if (len(row[1])==0):
                row[1] = None
            if (len(row[2]) == 0):
                row[2] = None
            record = CurrencyDetails(*row)
            s.add(record)
    s.commit()
    print('Done.')
    
def load_transactions(session, filename='train_transactions.csv'):
    #Retrieve country codes to standardise to one format
    country_dict = {}
    country_reader = csv.reader(open('countries.csv', 'r'))
    next(country_reader)  # Skip the header row.
    for line in country_reader:
        country_dict[line[0].upper()]=line[2].upper()

    session = sessionmaker()
    session.configure(bind=engine)
    s = session()
    print('Loading transactions...')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row.
        for row in reader:
            row = row[1:len(row)]
            if (row[5].upper() in country_dict.keys()):
                row[5] = country_dict[row[5].upper()]
            elif (row[5].upper() in country_dict.values()):
                row[5] = row[5].upper()
            else:
                row[5] = None
            record = Transactions(*row)
            s.add(record)
    s.commit()
    print('Done.')

def load_users(session, filename='train_users.csv'):
    #Retrieve fraudster list
    fraudster_list = []
    fraudster_reader = csv.reader(open('train_fraudsters.csv', 'r'))
    for line in fraudster_reader:
        fraudster_list.append(line[1])
    s = session()
    print('Loading users...')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row.
        
        for row in reader:
            if (row[10] in fraudster_list):
                is_fraudster = True
            else:
                is_fraudster = False
            if (len(row[7])==0):
                row[7] = None
            row[9] = True if row[9]=='1' else False
            row = [row[10], row[9], row[8], is_fraudster, row[7], row[6], row[5], row[4], row[3], row[2], row[1]]    
            record = Users(*row)
            s.add(record)
    s.commit()
    print('Done.')

def load_fxrates(session, filename='fx_rates.csv'):
    s = session()
    print('Loading fx rates...')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  
        for row in reader:
            for col in range(1, len(row)):
                row_temp = [row[0], header[col][0:3], header[col][3:6], row[col]]
                print(row_temp)
                record = FxRates(*row_temp)
                s.add(record)
    s.commit()
    print('Done.')


engine = create_engine(conn_str, echo=False)
session = sessionmaker()
session.configure(bind=engine)

load_currency_details(session)
load_transactions(session)
load_users(session)
load_fxrates(session)


 
