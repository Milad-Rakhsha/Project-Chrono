import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from decimal import Decimal

from scipy.optimize import curve_fit

def func(x, a):
    return a * np.power(-x,1.5)

def funcLinear(x, b):
    return b * x


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

d0x=[]
d1x=[]
with open('./ref_data/YangEtAl10X.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        d0x.append(float(row[0]))
        d1x.append(float(row[1]))
    

d0y=[]
d1y=[]    
with open('./ref_data/YangEtAl10Y.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        d0y.append(float(row[0]))
        d1y.append(float(row[1]))


os.chdir("/home/milad/CHRONO/buildProjectMiladFSI-FlexRelease/DEMO_OUTPUT/FSI_Validation")

c0=[]
c1=[]
c2=[]
c3=[]

with open('Analysis.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        c0.append(float(row[0]))
        c1.append(float(row[1]))
        c2.append(float(row[2]))
        c3.append(float(row[3]))



t=np.array(c0); #time
x=np.array(c1)-c1[0]; #x
y=np.array(c2)-c2[0]; #y
z=np.array(c3)-c3[0];#z
tx_exp=np.array(d0x); #t
x_exp=np.array(d1x)-d1x[0];#x
ty_exp=np.array(d0y); #t
y_exp=np.array(d1y)-d1y[0];#y

fig = plt.figure(num=None, figsize=(12, 6), dpi=140, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(211)
ax1.set_title("FSI Validation")    
ax1.set_ylabel('y($mm$)')
ax1.grid(color='k', linestyle='-', linewidth=0.2)
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.plot(t,z*1000, 'b', label='IISPH')
ax1.plot(ty_exp,y_exp*1000, 'or', label='Yang et al.')

ax2 = fig.add_subplot(212)
ax2.grid(color='k', linestyle='-', linewidth=0.2)
ax2.set_ylabel('x($mm$)')
ax2.set_xlabel("time(s)")
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.plot(t,x*1000, 'b')
ax2.plot(tx_exp,x_exp*1000, 'or', label='Yang et al.')

leg = ax1.legend()
plt.show()
