#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:40:19 2023

@author: joshuacarloapariciocasapao
"""

import numpy as np
import distillationExperiment as dx
import bellDiagEstimation as bd

from time import time

import matplotlib.pyplot as plt
import matplotlib.colors as pltco

plt.style.use('classic')

#%% Empirical parameters

## Depolarizing channel characteristic time
TdpoA = 100.
TdpoB = 100.

## Dephasing channel characteristic time
TdphA = 100.
TdphB = 100.

## Local rotations, noise parameters
mA = 0.01
mB = 0.01

## CNOT, noise parameters
yA = 0.01
yB = 0.01

## Z measurement, noise parameter
etaZA = 0.99
etaZB = 0.99

## X measurement, noise parameter
etaXA = 0.99
etaXB = 0.99

## Successful pair generation rate
pSuccess = 0.2                              ##typical expt'l parameter

## Total number of source-target pairs
totSamples1 = 2*10**5
totSamples2 = 2*10**5
totSamples3 = 2*10**5

#%% Generate tArray and q vectors

## fix seed value for replication
np.random.seed(8888)

## Time arrays
tArr1 = np.random.geometric(pSuccess, totSamples1)
tArr2 = np.random.geometric(pSuccess, totSamples2)
tArr3 = np.random.geometric(pSuccess, totSamples3)

## Convex set
q1Arr = np.linspace(0.5, 1.0, 75, endpoint=False)[1:]
q2Arr = np.linspace(0.0, 0.5, 75, endpoint=False)[1:]

q1Cx, q2Cx = np.meshgrid(q1Arr, q2Arr)    
convex = q1Cx + q2Cx <= 1.

## Array for the allowed values
q1Cx = q1Cx[convex]
q2Cx = q2Cx[convex]

## Generate the Bell vectors
## Here, we set the remaining Bell coefficients q3 = q4 = (1-q1-q2)/2
q3Cx = 0.5*(1.-q1Cx[...,None]-q2Cx[...,None])

qVecArr = np.hstack((q1Cx[...,None],q2Cx[...,None],q3Cx,q3Cx))

#%% Generate the empirical success probabilities

## fix seed value for replication
np.random.seed(8888)

## Empirical probabilities
pArrEmp = np.zeros((np.shape(qVecArr)[0],3))

for k in np.arange(np.shape(qVecArr)[0]):
    pArrEmp[k,0] = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                   qVecArr[k,:],qVecArr[k,:],tArr1,True)
    
    pArrEmp[k,1] = dx.protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                                   qVecArr[k,:],qVecArr[k,:],tArr2,True)
        
    pArrEmp[k,2] = dx.protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                   qVecArr[k,:],qVecArr[k,:],tArr3,True)

#%% Inversion

xVecArrEst = np.zeros((np.shape(qVecArr)[0],3))
qVecArrEst = np.zeros((np.shape(qVecArr)[0],4))

## Set the precision on x_i's
epsxArr = np.array([1e-2,1e-2,1e-2])

t = time()

for k in np.arange(np.shape(qVecArr)[0]):
    xVecArrEst[k,:] = bd.invertBellProtocolI(TdpoA,TdphA,TdpoB,TdphB,mA,mB,\
                                             yA,yB,etaZA,etaZB,etaXA,etaXB,\
                                             tArr1,tArr2,tArr3,pArrEmp[k,:],\
                                             epsxArr)
    
    qVecArrEst[k,:] = bd.convertToQ(xVecArrEst[k,:])
    
print(time()-t)

#%% Calculation of the trace distance

trDistArr = 0.5*np.sum(np.abs(qVecArrEst - qVecArr), axis=1)

"PLOTTING"
fig, ax = plt.subplots()
ax.set(aspect=1.0)

norm = pltco.TwoSlopeNorm(vmin=0.0, vcenter=np.sum(epsxArr), vmax=0.07)
#cmap = pltco.LinearSegmentedColormap.from_list("", ["blue","yellow","red"])
cmap = 'bwr'

sc = ax.scatter(q1Cx, q2Cx, c=trDistArr, s=25, marker='s', edgecolors='none',\
                cmap=cmap, norm=norm)
    
## make plot to reference Werner states
ax.plot([0.5,1.],[1./6.,0.], color='black', lw=2.5, ls='--')

## make the diagonal straight instead of jagged
## for cosmetic purposes only!
ax.plot([0.5+0.0037,1+0.0037],[0.5,0], color='white', lw=3.75, ls='-')

ax.set_xlim([0.5,1.0])
ax.set_ylim([0.0,0.5])

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

ax.set_xlabel(r'$q_1$', fontsize=36, labelpad=-3)
ax.set_ylabel(r'$q_2$', fontsize=36, labelpad=-3)
ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='y', which='major', length=8)

ax.annotate(r'$\epsilon_i = 10^{-2}$', \
            xy=(1.0,0.5), xytext=(0.82, 0.36), color='black',fontsize = 28)
    
ax.annotate(r'$q_3 = q_4 = \frac{1-q_1-q_2}{2}$', \
            xy=(1.0,0.5), xytext=(0.68, 0.44), color='black',fontsize = 28)

cbar = fig.colorbar(sc)
cbar.set_label(r'$\frac{1}{2}\|\hat{\mathbf{q}} -\mathbf{q}\|_1$', fontsize=36)
cbar.ax.tick_params(labelsize=18)
cbar.ax.set_yscale('linear')

fig.tight_layout() 

plt.savefig('distillation-bell-one-norm-distance-bwr.pdf', format = 'pdf', dpi = 300)

plt.show()

#%% Calculation of the failure probabilities

epspArr = np.zeros((np.shape(qVecArr)[0],3*2))

for k in np.arange(np.shape(qVecArr)[0]):
    epsArrTemp = np.zeros(3*2)
    
    for i in np.arange(3):
        xVecTempL     = np.copy(xVecArrEst[k,:]) 
        xVecTempL[i] -= epsxArr[i]
        
        xVecTempR     = np.copy(xVecArrEst[k,:])
        xVecTempR[i] += epsxArr[i]
        
        ## Convert to q vectors
        qVecTempL = bd.convertToQ(xVecTempL)
        qVecTempR = bd.convertToQ(xVecTempR)

        if i + 1 == 1: ## protocol 1
            rateEpsL =   dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                         qVecArrEst[k,:],qVecArrEst[k,:],tArr1,False) - \
                         dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                         qVecTempL,qVecTempL,tArr1,False) 
            
            rateEpsR = - dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                         qVecArrEst[k,:],qVecArrEst[k,:],tArr1,False) + \
                         dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                         qVecTempR,qVecTempR,tArr1,False)

        elif i + 1 == 2: ## Protocol 2
            rateEpsL =   dx.protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                                         qVecArrEst[k,:],qVecArrEst[k,:],tArr2,False) - \
                         dx.protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                                         qVecTempL,qVecTempL,tArr2,False)
            
            rateEpsR = - dx.protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                                         qVecArrEst[k,:],qVecArrEst[k,:],tArr2,False) + \
                         dx.protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                                         qVecTempR,qVecTempR,tArr2,False)
        
        else: ## Protocol 3
            rateEpsL =   dx.protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                         qVecArrEst[k,:],qVecArrEst[k,:],tArr3,False) - \
                         dx.protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                         qVecTempL,qVecTempL,tArr3,False)
            
            rateEpsR = - dx.protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                         qVecArrEst[k,:],qVecArrEst[k,:],tArr3,False) + \
                         dx.protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                         qVecTempR,qVecTempR,tArr3,False)
            
        epsArrTemp[i], epsArrTemp[i+3] = abs(rateEpsL), abs(rateEpsR)    
        
    epspArr[k,:] = epsArrTemp

#%% Calculate empirical Hoeffding bounds

delta =   np.longdouble(np.exp(-2.*np.array([totSamples1,totSamples2,totSamples3])*epspArr[:,0:3]**2.)) \
        + np.longdouble(np.exp(-2.*np.array([totSamples1,totSamples2,totSamples3])*epspArr[:,3:6]**2.))

deltaHoeffdingArr = np.longdouble(1. - np.prod(1. - delta, axis=1))

#%% Plots

"PLOTTING"
fig, ax = plt.subplots()
ax.set(aspect=1.0)

norm = pltco.TwoSlopeNorm(vmin=-9, vcenter=-2, vmax=0.)
#cmap = pltco.LinearSegmentedColormap.from_list("", ["blue","yellow","red"])
cmap = 'bwr'

sc = ax.scatter(q1Cx, q2Cx, c=np.log10(deltaHoeffdingArr), s=25, marker='s', \
                edgecolors='none',\
                cmap=cmap, norm=norm)
    
#make plot to reference Werner states
ax.plot([0.5,1.],[1./6.,0.], color='black', lw=2.5, ls='--')

#make the diagonal straight instead of jagged
#for cosmetic purposes only!
ax.plot([0.5+0.0037,1+0.0037],[0.5,0], color='white', lw=3.75, ls='-')

ax.set_xlim([0.5,1.0])
ax.set_ylim([0.0,0.5])

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

ax.set_xlabel(r'$q_1$', fontsize=36, labelpad=-3)
ax.set_ylabel(r'$q_2$', fontsize=36, labelpad=-3)
ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='y', which='major', length=8)

ax.annotate(r'$\epsilon_i = 10^{-2}$', \
            xy=(1.0,0.5), xytext=(0.82, 0.36), color='black',fontsize = 28)
    
ax.annotate(r'$q_3 = q_4 = \frac{1-q_1-q_2}{2}$', \
            xy=(1.0,0.5), xytext=(0.68, 0.44), color='black',fontsize = 28)

cbar = fig.colorbar(sc, extend='min')
    
cbar.set_label(r'$\log_{10}(\delta_{\mathrm{Hoeffding}})$', \
               fontsize=36)
    
cbar.ax.tick_params(labelsize=18)
cbar.ax.set_yscale('linear')

fig.tight_layout() 

plt.savefig('distillation-bell-failure-prob-bwr.pdf', format = 'pdf', dpi = 300)

plt.show()

#%% save txt file

result = np.hstack((qVecArr, pArrEmp, \
                    xVecArrEst, qVecArrEst,\
                    trDistArr[...,None],deltaHoeffdingArr[...,None]))

np.savetxt('belldiagonal_N2e5_Allepsx1e-2.txt', result, delimiter='\t', newline='\n',\
           header="qVectorTrue, pEmpirical, xVectorEstimated, qVectorEstimated, traceDistance, deltaHoeffding")



