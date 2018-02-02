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

F_exp=400e-3
delta_s_exp=100e-6;
K=F_exp/np.power(delta_s_exp,1.5)



os.chdir("/home/milad/CHRONO/buildProjectMiladFSI-FlexRelease/DEMO_OUTPUT/Indentation_test/Indentation_test")

c0=[]
c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
with open('Analysis.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        c0.append(float(row[0]))
        c1.append(float(row[1]))
        c2.append(float(row[2]))
        c3.append(float(row[3]))
        c4.append(float(row[4]))
        c5.append(float(row[5]))


t=np.array(c0);
ds=np.array(c1);
Ff=np.array(c2);  # fluid
shf=np.array(c3); # shell
Fi=np.array(c4);  # indentor
rho=np.array(c5);

Indentation_rate=(ds[1]-ds[0])/(t[1]-t[0])
fig = plt.figure(num=None, figsize=(12, 6), dpi=140, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax1.set_title("Indentation rate =%.1f $\mu m/s, $"%(Indentation_rate*1e6))    
ax1.set_ylabel('Force ($mN$)')
ax1.grid(color='k', linestyle='-', linewidth=0.2)
ax1.autoscale(enable=True, axis='x', tight=True)
popt, pcov = curve_fit(func, ds, Ff)
popt_linear, pcov_linear = curve_fit(funcLinear, ds, Ff)
ax1.plot(ds*-1e6,Ff*1e3, c='b', label='Fluid force       F=P * A')
N=50
ax1.plot(ds*-1e6, funcLinear(ds, *popt_linear)*1e3, 'ko', label='Linear fit          F=%.1e * $\delta_s $' % -popt_linear, ms=3, markevery=N)
ax1.plot(ds*-1e6, func(ds, *popt)*1e3, 'r', label='Non-linear fit   F=%.1e * $\delta_s^{1.5} $' % popt)
ax1.plot(ds*-1e6,np.power(-ds, 1.5)*1e3*4e5, c='g', label='Experiment      F=%.1e * $\delta_s ^{1.5}$' %K)
#ax1.plot(ds*-1e6,Fi*1e3, c='r', label='Indentor force')
#ax2 = fig.add_subplot(212)
#ax2.grid(color='k', linestyle='-', linewidth=0.2)
#ax2.set_xlabel('Indentor displacement ($\mu m$)')
#ax2.set_ylabel("Density Error(%)")
#ax2.autoscale(enable=True, axis='x', tight=True)
#plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f')) 
#ax2.plot(ds*-1e6,np.array((rho-1000)/10), c='b', label='fluid force')

leg = ax1.legend()
plt.show()
print('F=k*d^(3/2), k=%e' % Decimal(K))
print(popt)
