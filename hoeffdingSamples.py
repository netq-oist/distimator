"""
Calculating the minimum number of samples to successfully estimate
via the distillation protocols (noiseless case)

"""

import numpy as np

from sympy.solvers import nsolve
from sympy import Symbol, exp

import matplotlib.pyplot as plt
import matplotlib.colors as pltco

plt.style.use('classic')

#%% DEFINITIONS

## we use bisection method available in sympy

def nHoeffdingDistillation(delta, epsx, xArr):
    
    ## protocol 1
    epsL1 =  epsx[0]**2. + epsx[0]*(2.*xArr[0] - 1.)
    epsR1 = -epsx[0]**2. + epsx[0]*(2.*xArr[0] - 1.)
    
    ## protocol 2
    epsL2 =  epsx[1]**2. + epsx[1]*(2.*xArr[1] - 1.)
    epsR2 = -epsx[1]**2. + epsx[1]*(2.*xArr[1] - 1.)
    
    ## protocol 3
    epsL3 =  epsx[2]**2. + epsx[2]*(2.*xArr[2] - 1.)
    epsR3 = -epsx[2]**2. + epsx[2]*(2.*xArr[2] - 1.)
     
    n = Symbol('n')

    samples = nsolve(1.- (1.-exp(-2.*n*epsL1**2.)-exp(-2.*n*epsR1**2.))*\
                         (1.-exp(-2.*n*epsL2**2.)-exp(-2.*n*epsR2**2.))*\
                         (1.-exp(-2.*n*epsL3**2.)-exp(-2.*n*epsR3**2.)) \
                       - delta, n, (1e4,1e10), solver='bisect', prec=1, verify=False)
        
    return np.ceil(samples) 
         
def nHoeffdingTomography(delta, epsx):
     
    n = Symbol('n')

    samples = nsolve(1.- (1.-2.*exp(-2.*n*(epsx[0]/2.)**2.))*\
                         (1.-2.*exp(-2.*n*(epsx[1]/2.)**2.))*\
                         (1.-2.*exp(-2.*n*(epsx[2]/2.)**2.)) - delta, n,\
                            (1e4,1e10), solver='bisect', prec=1, verify=False)
        
    return np.ceil(samples)

#%% INPUT DATA

## Convex set
q1Arr = np.linspace(0.50, 1.00, 75, endpoint=False)[1:]
q2Arr = np.linspace(0.00, 0.50, 75, endpoint=False)[1:]

q1Cx, q2Cx = np.meshgrid(q1Arr, q2Arr)    
convex = q1Cx + q2Cx <= 1.

## Array for the allowed values
q1Cx = q1Cx[convex]
q2Cx = q2Cx[convex]

## Generate the Bell vectors
## Here, we set the remaining Bell coefficients q3 = q4 = (1-q1-q2)/2
q3Cx = 0.5*(1.-q1Cx[...,None]-q2Cx[...,None])

#qVecArr = np.hstack((q1Cx[...,None],q2Cx[...,None],q3Cx,q3Cx))

xVecArr = np.hstack((q1Cx[...,None]+q2Cx[...,None],q1Cx[...,None]+q3Cx,\
                     q1Cx[...,None]+q3Cx))

## Set the precision on x_i's
epsxArr = np.array([1e-2,1e-2,1e-2])    

## Set maximum failure probability
delta = 1e-2

#%% CALCULATE SUCCESS PROBABILITIES

protProbArr = 0.5*(xVecArr**2. + (1.-xVecArr)**2.)

#%% CALCULATE FOR THE TOTAL NUMBER SAMPLES

## FOR DISTILLATION
samplesArr = np.zeros(np.shape(xVecArr)[0])

for k in np.arange(np.shape(xVecArr)[0]):
    samplesArr[k] = nHoeffdingDistillation(delta, epsxArr, xVecArr[k,:])

## Adjust the values, to directly compare total states
## consumed in distillation, and that in tomography

## total consumed resources FOR TOMOGRAPHY
tomSamples = np.float64(nHoeffdingTomography(delta, epsxArr))    

## each distillation protocol needs 2 input states;
## all distillation protocols have the same total number of input pairs 
## of Bell pairs; total expected consumption
adjSamples = 3*samplesArr + np.sum(samplesArr[...,None]*(1.-protProbArr), axis=1)

## display the results 
print('Minimum number with distillation:', min(adjSamples))
print('Minimum number with tomography:', 3*tomSamples)
print('Maximum efficiency:', 1. - (min(adjSamples)/(3*tomSamples)))

#%% PLOTTING

"PLOTTING"

## we simply set so that plotting is manageable
vmin = 1e5
vc = 3*tomSamples
vmax = 1e6

adjSamples = np.where(adjSamples >=  vmax,  vmax, adjSamples)
adjSamples = np.where(adjSamples <=  vmin,  vmin, adjSamples)

fig, ax = plt.subplots()
ax.set(aspect=1.0)

norm = pltco.TwoSlopeNorm(vmin=vmin,vcenter=vc,vmax=vmax)
cmap = pltco.LinearSegmentedColormap.from_list("", ["blue","yellow","red"])

sc = ax.scatter(q1Cx, q2Cx, c=adjSamples, s=25, marker='s', \
                edgecolors='none',\
                cmap=cmap, norm=norm)

## make plot to reference Werner states
ax.plot([0.5,1.],[1./6.,0.], color='black', lw=2., ls='--')

ax.annotate(r'${\mathbf{q}}=\left({q}_1,\frac{1-{q}_1}{3},\frac{1-{q}_1}{3},\frac{1-{q}_1}{3}\right)$', \
            xy=(0.5,0.5), xytext=(0.53, 0.075), color='black',fontsize = 19,\
                rotation=-18.5)
    
## make the diagonal straight instead of jagged
## for cosmetic purposes only!
ax.plot([0.5+0.00475,1+0.00475],[0.5,0], color='white', lw=3.75, ls='-')

ax.set_title(r'$\delta_{\mathrm{Hoeffding}} = 10^{-2},\quad\epsilon_i = 10^{-2}$',\
             y=1.025, fontsize=24)

ax.set_xlim([0.5,1.0])
ax.set_ylim([0.0,0.5])

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

ax.set_xlabel(r'${q}_1$', fontsize=36, labelpad=-3)
ax.set_ylabel(r'${q}_2$', fontsize=36, labelpad=-3)
ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='y', which='major', length=8)
    
ax.annotate(r'${q}_3 = {q}_4 = \frac{1-{q}_1-{q}_2}{2}$', \
            xy=(1.0,0.5), xytext=(0.67, 0.44), color='black',fontsize = 28)

cbar = fig.colorbar(sc, ticks=[vmin, vc, vmax])
cbar.ax.set_yticklabels([r'$\leq 10^5$', \
                         r'$N_{\mathrm{tom}}\approx 3.8\times10^5$', \
                             r'$\geq 10^6$'])
cbar.ax.set_title(r'$N_{\mathrm{consumed}}$', fontsize=32, rotation=0, x=2.75, y=1.05)
cbar.ax.tick_params(labelsize=20)

fig.tight_layout() 

plt.savefig('belldiagonal_samples.png', format = 'png', dpi = 300)

plt.show()

#%% SAVE TXT FILE

qVecArr = np.hstack((q1Cx[...,None],q2Cx[...,None],q3Cx,q3Cx))

result = np.hstack((qVecArr, xVecArr, samplesArr[...,None], protProbArr, adjSamples[...,None]))

np.savetxt('belldiagonal_Samples_Allepsx1e-2_Delta1e-2.txt', result, \
           delimiter='\t', newline='\n',\
           header="qVector, xVector, total number of 2-copies, success probabilities, total copies consumed")




    





















