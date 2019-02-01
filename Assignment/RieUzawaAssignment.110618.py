#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:28:28 2018

@author: rie
"""

import pandas as pd

import numpy as np

data = pd.read_csv("PythonGroup/Brapa_cpm.csv")

data = np.array(data) 

data.shape #(100, 36)
data

# Q1: make a new array where the values are converted to log2 cpm

dat1 = data +1  ###I added 1 because there are 0s in array
np.log2(dat1) 

# Q2: Which gene has the highest expression?
data.argmax() #2992 so 2992/36 sample  = 83.1111. Gene #84 has the max gene exrepssion 


# Q3: Make a new array in which each row is sorted by expression
dat2 = np.sort(data)
dat2 = np.sort(data, axis = 1)


# Q4: Make a new array in which the samples are sorted according to average expression.  NOTE this is different than sorting each row...explain how

##### 1) get mean for each gene. 2) Substract the mean from each value in each row, 3) sort based on the substracted value
dat3 = data - data.mean(0)
dat3.sort(1)

# Q5: what is the total number of unique values in the table?
np.sum(np.unique(data))  ####104151.743163242


# Q6: Make a new array in which any cell that has less than 10 cpm is replaced with 0
dat4 = np.where(data <10, 0, data)

# Q7: Make a new array that only retains genes that are expressed at > 10 cpm in at least half the samples.  (I haven't checked, maybe all genes will pass)

dat4.sort(1) #### sort each row by ascending order, using the previous array 
dat5 = dat4[dat4[:,18]>0] ###Keep the rows with more than half samples are more than 10 cpm by looking if the 
dat5

# Q8 calculate the standard deviation for each gene using the builtin function
data.std(axis = 1)

# Q9 calcualte the standard deviation for each gene WITHOUT using the builtin function

####I could not figure out

# Q10 create a slice that contains the first 10 samples
data[:,:10]   

# Q11 Changing topics...generate and plot a 2D random walk (see Chapter 4.7)

import random
import matplotlib.pyplot as plt

position = 0
walk = [position]
steps = 100
for i in range(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
    
plt.plot(walk)
