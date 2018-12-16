#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:00:17 2018
Helper functions needed along the way of analysing the data and solving the challenge
@author: dhirensarin
"""

from sqlalchemy.inspection import inspect
import pandas as pd

#Helper function to convert ORM object to list
def orm_to_df(rset):
    """List of result
    Return: columns name, list of result
    """
    result = []
    for obj in rset:
        instance = inspect(obj)
        items = instance.attrs.items()
        result.append([x.value for _,x in items])
    column_names = instance.attrs.keys()
    data = result
    return pd.DataFrame(data, columns=column_names)