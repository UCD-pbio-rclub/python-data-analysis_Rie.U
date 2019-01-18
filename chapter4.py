i#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:30:44 2018

@author: rie
"""
import numpy as np

my_arr =  np.arange(1000000)
my_list = list(range(1000000))

%time for _ in range(10): my_arr2 = my_arr * 2

%time for _ in range (10) : my_list2 = [x * 2 for x in my_list]


data = np.random.randn(2,3)
data

data * 10

data + data 
 
data.shape

data.dtype

data1 = [6, 7.5, 8, 0, 1]

arr1 = np.array(data1)

data2 = [[1,2,3,4],[5,6,7,8]]

arr2 = np.array(data2)

arr2

arr2.ndim

arr2.shape

arr2.dtype

np.zeros((4,8,3))

np.arange(12)


arr1 = np.array([1,2,3], dtype = np.float64)
arr2 = np.array([1,2,3], dtype = np.int32)


arr =  np.array([1,2,3,4,5])
arr.dtype


float_arr = arr.astype(np.float64)
float_arr.dtype


arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8]=12
arr
 arr_slice = arr[5:8]
arr_slice

arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr


arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr2d
arr2d[2]
arr2d[0][2]

arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d
arr3d[0]
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d
arr3d[1][0]
x = arr3d[1]
x
x[0]
 arr
arr[1:6]
arr2d
arr2d[:2]
arr2d[1:2]

names = np.array(['Bob','Joe', 'will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
names
data
names == 'Bob'
data[names == 'Bob']
data[names == 'Joe', 1:]
data[names == 'Joe', 2]
data[~(names == 'Joe')]
dat[names!=='Joe']
data[data<0]=0
data

arr = np.arange(15).reshape((3,5))
arr
arr.T

arr = np.random.randn(6, 3)
arr
np.dot(arr.T, arr)

arr.T



########################################################################
######4-3

points = np.arange (-5,5,0.01)
xs, ys = np.meshgrid(points, points)
ys
xs
z = np.sqrt(xs ** 2 + ys ** 2)
z

import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()



xarr = np.array([1.1,1.2,1.3,1.4,1.5])

yarr = np.array([2.1,2.2,2.3,2.4,2.5])

cond = np.array([True, False,True, True, False])

result = [(x if c else y)
            for x, y, c in zip(xarr,yarr,cond)]

result #[1.1, 2.2, 1.3, 1.4, 2.5]


result = np.where(cond, xarr, yarr)

arr = np.random.randn(4,4)
arr
arr>0
np.where(arr>0, 2, -2)
np.where(arr>0, 2, arr)


arr = np.random.randn(5,4)
arr
arr.mean()
np.mean(arr)
arr.sum()

arr.mean(axis = 1)
1.29992205-0.61915665+0.39494264+1.61708267-0.08033327
arr.sum(axis = 0)


arr = np.random.randn(10)
arr
(arr>0).sum()


arr = np.random.randn(6)
arr
arr.sort()
arr


arr = np.random.randn(5,3)
arr
arr.sort(0)
arr

from numpy.linalg import inv, qr
x = np.random.randn(5,5)
x
mat = x.T.dot(x)


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
plt.plot(walk[:100])


nsteps = 1000
draws = np.random.randint(0,2, size = nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()
walk.max()
 
(np.abs(walk) >=10).argmax()

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks
walks.max()
hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()
