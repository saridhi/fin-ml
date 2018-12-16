#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:38:47 2018

The approach to this algorithm is to create a nXn matrix of exchange rates.
Following this, we need to discover all paths of conversions, essentially
in a tree-like manner. This is done by finding all possible permutations for 
length 0 to n. These permutation indices are then referenced sequentially
to perform multiplication operations. Inconsistencies are discovered when the 
multiplied results > 1 as this represents an arbitrage opportunity. On the first
inconsistency that is found, the program exits.

@author: dhirensarin
"""
import csv
import numpy as np
import itertools

#helper function to find permutations to build a tree of conversion choices
def get_permutations(mat_len):
    temp = list(range(0, mat_len))
    #This only deals with currencies starting and ending with the first row
    temp.extend([0])
    temp = list(itertools.permutations(temp))  
    #only select permutations that return to base currency
    ret_set = set([i for i in temp if i[0]==i[len(i)-1]])
    return ret_set

#Find all possible permutations essentially building a tree of conversion choices
def all_permutations(mat_len):
    s = {*()} 
    for i in range(2, mat_len+1):
        s.update(get_permutations(i))
    return s
    
def is_arbitrage(table):
    i = 0
    while i<len(table):
        #Retrieve matrix size from first row of table
        mat_size = int(table[i][0])
        #Create matrix without the ones on the diagonal
        mat = np.matrix(table[i+1:i+1+mat_size], dtype=float)
        #Create new matrix with ones on the diagonal
        new_mat = np.ones((mat_size, mat_size), dtype=float)
        #Fill the new matrix with the sample data
        for row in range(0, len(mat)):
            new_mat[row] = np.insert(mat[row], row, 1)
        #Default to no arbitrage
        flag = False
        #Iterate through all permutations
        for s in all_permutations(mat_size):
            total_pnl = 0
            #Pinpoint locations in the matrix to multiply
            for n in range(0, len(s)-1, 2):
                x1=s[n]
                y1=s[n+1]
                x2=s[n+1]
                #If odd number of elements, ignore the last element multiplication
                if n+2==len(s):
                    pnl = new_mat[x1,y1]
                else:
                    y2=s[n+2]
                    pnl = new_mat[x1,y1]*new_mat[x2,y2]
                #Keep track of total pnl if > 1 for arbitrage condition
                total_pnl = total_pnl + pnl
            if (total_pnl > 1):
                #Found arb opportunity!
                print([x+1 for x in s])
                flag = True
                break
        if (flag==False):
            print("no arbitrage sequence exists")
        i = i + mat_size + 1
    return flag


#Reading the sample file with the example pasted from the question
with open('sample_2.txt', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    table = list(csv_reader)

#Run the arbitrage function
is_arbitrage(table)

