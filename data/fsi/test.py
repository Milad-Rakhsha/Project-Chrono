#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:48:26 2018

@author: milad
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

u=np.linspace(-2, 2.0, num=10000)
I=np.log(-u)
fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='b')
ax1 = fig.add_subplot(111)

for t in np.array([0.001,0.01 ,0.1 ,1, 10]):
    ax1.plot(u,-I/t,label="t={}".format(t))
    
    
ax1.set_xlabel('u', fontsize=20)   
ax1.set_ylabel("$\hat{I}(u)=-1/t log(-u)$", fontsize=20)    
 
ax1.grid(color='k', linestyle='-', linewidth=0.1)    
plt.legend(prop={'size': 16})
plt.show()