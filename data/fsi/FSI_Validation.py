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

yangx0=[]
yangx1=[]
with open('./ref_data/YangEtAl10X.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        yangx0.append(float(row[0]))
        yangx1.append(float(row[1]))
    

yangy0=[]
yangy1=[]    
with open('./ref_data/YangEtAl10Y.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        yangy0.append(float(row[0]))
        yangy1.append(float(row[1]))

antocix0=[]
antocix1=[]
with open('./ref_data/Antoci_x.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        antocix0.append(float(row[0]))
        antocix1.append(float(row[1]))
    

antociy0=[]
antociy1=[]    
with open('./ref_data/Antoci_y.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        antociy0.append(float(row[0]))
        antociy1.append(float(row[1]))

exp_x0=[]
exp_x1=[]
with open('./ref_data/Ex10X.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        exp_x0.append(float(row[0]))
        exp_x1.append(float(row[1]))
    

exp_y0=[]
exp_y1=[]    
with open('./ref_data/Ex10Y.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        exp_y0.append(float(row[0]))
        exp_y1.append(float(row[1]))


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
tx_yang=np.array(yangx0); #t
x_yang=np.array(yangx1)-yangx1[0];#x
ty_yang=np.array(yangy0); #t
y_yang=np.array(yangy1)-yangy1[0];#y

tx_antoci=np.array(antocix0); #t
x_antoci=np.array(antocix1)-antocix1[0];#x
ty_antoci=np.array(antociy0); #t
y_antoci=np.array(antociy1)-antociy1[0];#y

tx_exp=np.array(exp_x0); #t
x_exp=np.array(exp_x1)-exp_x1[0];#x
ty_exp=np.array(exp_y0); #t
y_exp=np.array(exp_y1)-exp_y1[0];#y

major_ticks = np.arange(0, 101, 20)
minor_ticks = np.arange(0, 101, 5)




fig = plt.figure(num=None, figsize=(8, 6), dpi=140, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(211)
ax1.set_title("FSI Validation")    
ax1.set_ylabel('y($mm$)')
ax1.grid(color='k', linestyle='-', linewidth=0.2)
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.plot(t,z*1000, 'k', label='IISPH')
ax1.plot(ty_antoci,y_antoci*1000, 'k--', label='Antoci et al')
ax1.plot(ty_exp,y_exp*1000, 'ko', label='Experimental')

ax1.set_xticks(np.linspace(0, 0.4, 9))
ax1.set_yticks(np.linspace(0, 20, 11))
ax1.grid(which='both', linestyle='--', linewidth=0.5)
ax1.set_xlim(0, 0.4)
ax1.set_ylim(0, 18)
ax1.set_yticks(np.linspace(0, 20, 21), minor=True)
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.5)



ax2 = fig.add_subplot(212)
ax2.grid(color='k', linestyle='-', linewidth=0.2)
ax2.set_ylabel('x($mm$)')
ax2.set_xlabel("time(s)")
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.plot(t,x*1000, 'k')
ax2.plot(tx_antoci,x_antoci*1000, 'k--', label='Antoci et al')
ax2.plot(tx_exp,x_exp*1000, 'ok', label='Experimental')

leg = ax1.legend()
ax2.set_xticks(np.linspace(0, 0.4, 9))
ax2.set_yticks(np.linspace(0, 50, 11))
ax2.grid(which='both', linestyle='--', linewidth=0.5)
ax2.set_xlim(0, 0.4)
ax2.set_ylim(0, 20)
ax2.set_yticks(np.linspace(0, 50, 21), minor=True)
ax2.grid(which='minor', alpha=0.2)
ax2.grid(which='major', alpha=0.5)
plt.show()
