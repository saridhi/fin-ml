#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:57:32 2018

Settings for the DB connection
@author: dhirensarin
"""

#Change this to your db details
db_password = 'hermitage'
db_name = 'postgres'
db_user = 'postgres'
db_host = 'localhost'
db_port = 5432

conn_str = "postgres://%s:%s@%s:%s/%s" % (db_user, db_password, db_host, db_port, db_name)
