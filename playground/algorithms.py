#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:52:09 2019

@author: dhirensarin
"""

def merge_arrays(a, b):
    for i,j in zip(a,b):
        print(i++)
        print(j)
    
print(merge_arrays([1,3,8], [2,5,7,10]))