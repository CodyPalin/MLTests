#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:25:12 2021

@author: codypalin
"""

# 2d indexing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
data2d = array([[11, 22, 33],
		[44, 55, 66],
		[77, 88, 99]])
# index data
print(data2d[:,2])