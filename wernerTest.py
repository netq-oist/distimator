#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:39:34 2023

@author: joshuacarloapariciocasapao
"""


import numpy as np
import distillationExperiment as dx
import wernerEstimation as we

import matplotlib.pyplot as plt
plt.style.use('classic')

#%% EMPIRICAL PARAMETERS

# Depolarizing channel characteristic time
TdpoA = 100.
TdpoB = 100.

# Dephasing channel characteristic time
TdphA = 100.
TdphB = 100.

# CNOT, noise parameters
yA = 0.01
yB = 0.01

# Z measurement, noise parameter
etaZA = 0.99
etaZB = 0.99

# Successful pair generation rate
pSuccess = 0.2                              ##typical expt'l parameter

# Total number of source-target pairs
totSamples5 = 10**5
totSamples6 = 10**6

#%% Generate tArray and Bell vectors

np.random.seed(8888)
tArray5 = np.random.geometric(pSuccess, totSamples5)
tArray6 = np.random.geometric(pSuccess, totSamples6)

# Initialize Bell vectors
wArray = np.linspace(0.0,0.66,20,False)
wvecArr = np.hstack([1-0.75*wArray[...,None],0.25*wArray[...,None],\
                       0.25*wArray[...,None],0.25*wArray[...,None]])

rateExpArr5 = np.zeros(len(wArray))
rateExpArr6 = np.zeros(len(wArray))
    
for k in np.arange(len(wArray)):
    # Empirical rate (simulated)
    rateExpArr5[k] = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                     wvecArr[k,:],wvecArr[k,:],tArray5,True) 
        
    rateExpArr6[k] = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                     wvecArr[k,:],wvecArr[k,:],tArray6,True)  
    
#%% INVERSION    

wInvArr5 = np.zeros(len(wArray))    
wInvArr6 = np.zeros(len(wArray))    

# Set the tolerance for the Werner parameter
epsw = 10.**-2

for k in np.arange(len(wArray)):
    # Werner parameter obtained via inversion
    wInvArr5[k] = we.invertWernerParamI(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                        tArray5,rateExpArr5[k],epsw)
        
    wInvArr6[k] = we.invertWernerParamI(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                        tArray6,rateExpArr6[k],epsw)

#%% CALCULATE LEFT AND RIGHT ERRORS ON RATE

## Evaluate for the tolerances for the heralding rates
wInvLeft5 = wInvArr5 - epsw
wInvRight5 = wInvArr5 + epsw

rateErr5 = np.zeros(len(wArray))

wInvLeft6 = wInvArr6 - epsw
wInvRight6 = wInvArr6 + epsw

rateErr6 = np.zeros(len(wArray))

for k in np.arange(len(wArray)):
  wLvec = np.array([1.-0.75*wInvLeft5[k],0.25*wInvLeft5[k],\
                       0.25*wInvLeft5[k],0.25*wInvLeft5[k]])
  wRvec = np.array([1.-0.75*wInvRight5[k],0.25*wInvRight5[k],\
                       0.25*wInvRight5[k],0.25*wInvRight5[k]])

  rateL = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                          wLvec,wLvec,tArray5,False)

  rateR = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                          wRvec,wRvec,tArray5,False)

  rateErr5[k] = max(abs(rateL - rateExpArr5[k]), abs(rateR - rateExpArr5[k]))
  
for k in np.arange(len(wArray)):
  wLvec = np.array([1.-0.75*wInvLeft6[k],0.25*wInvLeft6[k],\
                       0.25*wInvLeft6[k],0.25*wInvLeft6[k]])
  wRvec = np.array([1.-0.75*wInvRight6[k],0.25*wInvRight6[k],\
                       0.25*wInvRight6[k],0.25*wInvRight6[k]])

  rateL = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                          wLvec,wLvec,tArray6,False)

  rateR = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                          wRvec,wRvec,tArray6,False)

  rateErr6[k] = max(abs(rateL - rateExpArr6[k]), abs(rateR - rateExpArr6[k]))

#%% CURVE FITS FOR RATES

## curve fit for the success probabilities
wFit = np.linspace(0.0,2/3.,100, False)
rateFit5 = np.zeros(len(wFit))
rateFit6 = np.zeros(len(wFit))

for q in np.arange(len(wFit)):
    wvec = np.array([1.-0.75*wFit[q],0.25*wFit[q],0.25*wFit[q],0.25*wFit[q]])
    
    rateFit5[q] = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                  wvec,wvec,tArray5,False)
    rateFit6[q] = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                                  wvec,wvec,tArray6,False)
      
#%% PLOTS

## Plot the experimental heralding rates
plt.style.use('classic')

fig, ax = plt.subplots()
ax.errorbar(wArray, rateExpArr5, yerr=rateErr5, fmt='o', ls='none',\
            label=r'$\hat{p}^{(1)}, \quad N^{(1)} = 10^{5}, \quad \epsilon_w = 10^{-2}$',\
                color='tab:blue',elinewidth=2,capsize=1,markersize=7)
ax.plot(wFit,rateFit5, ls='--',color='red',zorder=0,linewidth=2.5)

ax.set_xlabel(r'$w$', fontsize=36)
ax.set_ylabel(r'${p}^{(1)}$', fontsize=36)

ax.set_xlim([0.6- 2/3.,2/3.])
ax.set_ylim([1/4.,1/2.])

ax.legend(fontsize=23, numpoints=1,loc='lower left')

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='y', which='major', length=8)

fig.tight_layout() 

left, bottom, width, height = [0.615, 0.575, 0.34, 0.34]

axinset = fig.add_axes([left, bottom, width, height])

axinset.errorbar(wArray, wInvArr5-wArray, yerr=[epsw]*len(wArray), fmt='o', \
            ls='none',color='darkorange')

axinset.plot([0.6- 2/3.,2/3.],[0.0,0.0],ls='--',color='black')    

axinset.set_xlabel(r'$w$', fontsize=24)
axinset.set_ylabel(r'$\hat{w} - w$', fontsize=24)    

axinset.tick_params(axis='y', labelsize=12.5)
axinset.tick_params(axis='x', labelsize=12.5)

axinset.set_xlim([0.6- 2/3.,2/3.])
axinset.set_ylim([-0.025,0.025])

plt.savefig('werner_errors_105.png', format = 'png', dpi = 300)

plt.show()

#%% PLOTS

## Plot the experimental success probabilities
plt.style.use('classic')

fig, ax = plt.subplots()
ax.errorbar(wArray, rateExpArr6, yerr=rateErr6, fmt='o', ls='none',\
            label=r'$\hat{p}^{(1)}, \quad N^{(1)} = 10^{6}, \quad \epsilon_w = 10^{-2}$',\
                color='tab:blue',elinewidth=2,capsize=1,markersize=7)
ax.plot(wFit,rateFit6, ls='--',color='red',zorder=0,linewidth=2.5)

ax.set_xlabel(r'$w$', fontsize=36)
ax.set_ylabel(r'${p}^{(1)}$', fontsize=36)

ax.set_xlim([0.6- 2/3.,2/3.])
ax.set_ylim([1/4.,1/2.])

ax.legend(fontsize=23, numpoints=1,loc='lower left')

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='y', which='major', length=8)

fig.tight_layout() 

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.615, 0.575, 0.34, 0.34]

axinset = fig.add_axes([left, bottom, width, height])

axinset.errorbar(wArray, wInvArr6-wArray, yerr=[epsw]*len(wArray), fmt='o', \
            ls='none',color='darkorange')

axinset.plot([0.6- 2/3.,2/3.],[0.0,0.0],ls='--',color='black')    

axinset.set_xlabel(r'$w$', fontsize=24)
axinset.set_ylabel(r'$\hat{w} - w$', fontsize=24)    

axinset.tick_params(axis='y', labelsize=12.5)
axinset.tick_params(axis='x', labelsize=12.5)

axinset.set_xlim([0.6- 2/3.,2/3.])
axinset.set_ylim([-0.025,0.025])

plt.savefig('werner_errors_106.png', format = 'png', dpi = 300)

plt.show()

#%% HOEFFDING FAILURE PROBABILITY (NOISELESS)

"Noiseless case fit"

epsw = 10.**-2

wArrayFit = np.linspace(0.0,2/3.,100, False)

deltanoise105 = 2*np.exp((-1/8.)*(10**5)*(-epsw**2. + 2.*epsw*(1.-wArrayFit))**2.)
deltanoise106 = 2*np.exp((-1/8.)*(10**6)*(-epsw**2. + 2.*epsw*(1.-wArrayFit))**2.)


#%% HOEFFDING FAILURE PROBABILITY (WITH GIVEN NOISE PARAMETERS)

"Expected maximum failure probability"

epsRateExpected5 = np.zeros(len(wArrayFit))
epsRateExpected6 = np.zeros(len(wArrayFit))

for k in np.arange(len(wArrayFit)):
    wvec = np.array([1.-0.75*wArrayFit[k],0.25*wArrayFit[k],\
                        0.25*wArrayFit[k],0.25*wArrayFit[k]])
    
    rate = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                           wvec,wvec,tArray5,False) 
        
    wL = wArrayFit[k] - epsw
    wR = wArrayFit[k] + epsw
    
    wLvec = np.array([1.-0.75*wL,0.25*wL,\
                         0.25*wL,0.25*wL])
    wRvec = np.array([1.-0.75*wR,0.25*wR,\
                         0.25*wR,0.25*wR])
    
    rateL = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                           wLvec,wLvec,tArray5,False)

    rateR = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                           wRvec,wRvec,tArray5,False)
    
    epsRateExpected5[k] = max(abs(rateL - rate), abs(rateR - rate))
    
for k in np.arange(len(wArrayFit)):
    wvec = np.array([1.-0.75*wArrayFit[k],0.25*wArrayFit[k],\
                        0.25*wArrayFit[k],0.25*wArrayFit[k]])
    
    rate = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                           wvec,wvec,tArray6,False) 
        
    wL = wArrayFit[k] - epsw
    wR = wArrayFit[k] + epsw
    
    wLvec = np.array([1.-0.75*wL,0.25*wL,\
                         0.25*wL,0.25*wL])
    wRvec = np.array([1.-0.75*wR,0.25*wR,\
                         0.25*wR,0.25*wR])
    
    rateL = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                           wLvec,wLvec,tArray6,False)

    rateR = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                           wRvec,wRvec,tArray6,False)
    
    epsRateExpected6[k] = max(abs(rateL - rate), abs(rateR - rate))

deltaRateExpected105 = 2.*np.exp(-2*(10**5)*epsRateExpected5*epsRateExpected5)
deltaRateExpected106 = 2.*np.exp(-2*(10**6)*epsRateExpected6*epsRateExpected6)

#%% simulated failure probability

deltaRate5 = 2.*np.exp(-2*len(tArray5)*rateErr5*rateErr5)
deltaRate6 = 2.*np.exp(-2*len(tArray6)*rateErr6*rateErr6)

#%% PLOTS

"Maximum failure probability plot"

fig, ax = plt.subplots()
    
ax.scatter(wArray, np.log10(deltaRate5), marker='o', \
           label=r'$N^{(1)} = 10^{5}$', color='darkorange', s=32)
    
ax.plot(wArrayFit, np.log10(deltanoise105), ls='-',color='darkorange', lw=1.75)
ax.plot(wArrayFit, np.log10(deltaRateExpected105), ls='--',color='darkorange', lw=1.75)
    
ax.scatter(wArray, np.log10(deltaRate6), marker='o', \
           label=r'$N^{(1)} = 10^{6}$', color='darkblue', s=32)
    
ax.plot(wArrayFit, np.log10(deltanoise106), ls='-',color='darkblue', lw=1.75)
ax.plot(wArrayFit, np.log10(deltaRateExpected106), ls='--',color='darkblue', lw=1.75)

ax.set_xlabel(r'$w$', fontsize=36)
ax.set_ylabel(r'$\log_{10}\,\delta$', fontsize=36)

ax.set_xlim([0.6- 2/3.,2/3.])
ax.set_ylim(top=0.)

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

ax.set_yscale('symlog', subs=[2,3,4,5,6,7,8,9])

ax.legend( bbox_to_anchor = (0.95,0.825), fontsize=24, numpoints=1, scatterpoints=1)

ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='y', which='major', length=8)
ax.tick_params(axis='y', which='minor', length=4.5)

fig.tight_layout() 

plt.savefig('werner_failure_probability_withref.png', format = 'png', dpi = 300)

plt.show()

#%% SAVE TXT FILE

result5 = np.hstack((wArray[...,None],rateExpArr5[...,None],\
                    wInvArr5[...,None],rateErr5[...,None],deltaRate5[...,None]))
result6 = np.hstack((wArray[...,None],rateExpArr6[...,None],\
                    wInvArr6[...,None],rateErr6[...,None],deltaRate6[...,None]))

np.savetxt('werner_N1e5_Allepsw1e-2.txt', result5, delimiter='\t', newline='\n',\
           header="w, pEmpirical, wEstimated, pEstimatedErr, deltaHoeffding")
    
np.savetxt('werner_N1e6_Allepsw1e-2.txt', result6, delimiter='\t', newline='\n',\
           header="w, pEmpirical, wEstimated, pEstimatedErr, deltaHoeffding")
    
#%% SAVE TXT FILE OF FITS

resultFit5 = np.hstack((wFit[...,None],rateFit5[...,None],\
                        deltanoise105[...,None],deltaRateExpected105[...,None]))
resultFit6 = np.hstack((wFit[...,None],rateFit6[...,None],\
                        deltanoise106[...,None],deltaRateExpected106[...,None]))

np.savetxt('wernerFit_N1e5_Allepsw1e-2.txt', result5, delimiter='\t', newline='\n',\
           header="w, p, deltaHoeffdingNoiseless, deltaHoeffdingWithNoise")
    
np.savetxt('wernerFit_N1e6_Allepsw1e-2.txt', result6, delimiter='\t', newline='\n',\
           header="w, p, deltaHoeffdingNoiseless, deltaHoeffdingWithNoise")

    
    
