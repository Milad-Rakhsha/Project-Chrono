import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import csv
import os
os.chdir("/home/milad/CHRONO/buildProjectMiladFSI-FlexRelease/DEMO_OUTPUT/Indentation_test/Indentation_test")

c0=[]
c1=[]
c2=[]
c3=[]
c4=[]
with open('Analysis.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        c0.append(float(row[0]))
        c1.append(float(row[1]))
        c2.append(float(row[2]))
        c3.append(float(row[3]))
        c4.append(float(row[4]))

t=np.array(c0);
ds=np.array(c1);
Ff=np.array(c2);
Fi=np.array(c3);
rho=np.array(c4);
Indentation_rate=(ds[2]-ds[1])/(t[2]-t[1])
fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(211)
ax1.set_title("Indentation rate ={} $\mu m/s$".format(Indentation_rate*1e6))    
ax1.set_ylabel('Force ($\mu N$)')
ax1.grid(color='k', linestyle='-', linewidth=0.2)
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.plot(ds*-1e6,Ff*1e3, c='b', label='Fluid force')
ax1.plot(ds*-1e6,Fi*1e3, c='r', label='Indentor force')
ax2 = fig.add_subplot(212)
ax2.grid(color='k', linestyle='-', linewidth=0.2)
ax2.set_xlabel('Indentor displacement ($\mu m$)')
ax2.set_ylabel("Density Error(%)")
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.plot(ds*-1e6,np.array((rho-1000)/10), c='b', label='fluid force')

leg = ax1.legend()
plt.show()
