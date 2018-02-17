#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:50:36 2018

@author: milad
"""
import numpy as np
y=np.array([1,2,3,4,5])
A= np.outer(np.transpose(y),y)
print(A)
