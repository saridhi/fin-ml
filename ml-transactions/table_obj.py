#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:06:50 2018
ORM objects used to create tables and load data
@author: dhirensarin
"""

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, Boolean, String, BigInteger, Numeric, DateTime
from sqlalchemy.ext.declarative import declarative_base
from settings import conn_str
from sqlalchemy_utils import UUIDType


engine = create_engine(conn_str, echo=True)
Base = declarative_base() 

########################################################################

class CurrencyDetails(Base):

    __tablename__ = "currency_details"

    ccy = Column(String(10), primary_key=True)
    iso_code = Column(Integer)
    exponent = Column(Integer)
    is_crypto = Column(Boolean, nullable=False)

    #----------------------------------------------------------------------

    def __init__(self, ccy, iso_code, exponent, is_crypto):

        """"""
        self.ccy = ccy
        self.iso_code = iso_code
        self.exponent = exponent
        self.is_crypto = is_crypto
        
class Users(Base):

    __tablename__ = "users"

    id = Column(UUIDType, primary_key=True)
    has_email = Column(Boolean, nullable=False)
    phone_country = Column(String(300))
    is_fraudster = Column(Boolean, nullable=False)
    terms_version = Column(Date)
    created_date = Column(DateTime(timezone=False), nullable=False)
    state = Column(String(25), nullable=False)
    country = Column(String(2))
    birth_year = Column(Integer)
    kyc = Column(String(20))
    failed_sign_in_attempts = Column(Integer)

    #----------------------------------------------------------------------

    def __init__(self, id, has_email, phone_country, is_fraudster, terms_version,
                 created_date, state, country, birth_year, kyc, failed_sign_in_attempts):

        """"""
        self.id = id
        self.has_email = has_email
        self.phone_country = phone_country
        self.is_fraudster = is_fraudster
        self.terms_version = terms_version
        self.created_date = created_date
        self.state = state
        self.country = country
        self.birth_year = birth_year
        self.kyc = kyc
        self.failed_sign_in_attempts = failed_sign_in_attempts


class Transactions(Base):

    __tablename__ = "transactions"

    currency = Column(String(3), nullable = False)
    amount = Column(BigInteger, nullable = False)
    state = Column(String(25), nullable = False)
    created_date = Column(DateTime(timezone=False), nullable=False)
    merchant_category = Column(String(100))
    merchant_country = Column(String(3))
    entry_method = Column(String(4))
    user_id = Column(UUIDType, nullable = False)
    type = Column(String(20), nullable = False) 
    source = Column(String(20), nullable = False)
    id = Column(UUIDType, primary_key = True)

    #----------------------------------------------------------------------

    def __init__(self, currency, amount, state, created_date, merchant_category,
                 merchant_country, entry_method, user_id, type, source, id):
   
        """"""
        self.currency = currency
        self.amount = amount
        self.state = state
        self.created_date = created_date
        self.merchant_category = merchant_category
        self.merchant_country = merchant_country
        self.entry_method = entry_method
        self.user_id = user_id
        self.type = type
        self.source = source
        self.id = id

class FxRates(Base):

    __tablename__ = "fx_rates"

    ts = Column(DateTime(timezone=False), primary_key=True)
    base_ccy = Column(String(3), primary_key=True)
    ccy = Column(String(10), primary_key=True)
    rate = Column(Numeric)
  
    #----------------------------------------------------------------------

    def __init__(self, ts, base_ccy, ccy, rate):
        """"""
        self.ts = ts
        self.base_ccy = base_ccy
        self.ccy = ccy
        self.rate = rate

# create tables
Base.metadata.create_all(engine)