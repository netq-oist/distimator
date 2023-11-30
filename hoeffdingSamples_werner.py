"""
Calculating the minimum number of samples to successfully estimate
via the distillation protocols (noiseless case)

"""

import numpy as np
import matplotlib.pyplot as plt 

plt.style.use('classic')
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

#%% Werner parameter vs samples

" noiseless case "

def sample_need_dist(delta, epsw, w):
    a = 8. * (1./(epsw**2. + 2.*epsw*(1.-w))**2.) * np.log(2./delta)
    return a ## counts the total number of pairs of states rho \otimes rho

" white noise "

def sample_need_dist_white(delta, epsw, w, S):
    a = 8. * (1./S**2.) * (1./(2.*epsw*(1.-w)+epsw**2.)**2.) * np.log(2./delta)
    return a ## counts the total number of pairs of states rho \otimes rho

" tomography "

def sample_need_tomography(delta, epsw):
    a = 8. * (1./((epsw)**2.)) * np.log(2./delta)
    return a ## counts the total number of stats

#%% Calculate the success probabilities

## Werner parameter range
wRange = np.linspace(0.,2/3.,1000, endpoint=False)

## set probability delta falling outside p^{(1)} threshold
delta = 0.01

## set value for the average depolarization S
S = np.exp(-1/4.)

" noiseless distillation "
p1Arr = 0.25*(2. - 2.*wRange + wRange**2.)

" distillation with white noise "
p1ArrWhite = 0.25*(S*wRange**2. - 2.*S*wRange + S + 1)

#%% Plotting

" noiseless distillation case "
w_dist = np.vectorize(sample_need_dist)

plt.plot(wRange, (2. - p1Arr)*w_dist(delta, 1e-2, wRange), label = r'$\epsilon_w = 10^{-2}$', \
         color='tab:blue',  lw=1.75)
plt.plot(wRange, (2. - p1Arr)*w_dist(delta, 1e-3, wRange), label = r'$\epsilon_w = 10^{-3}$', \
         color='tab:red',   lw=1.75)
plt.plot(wRange, (2. - p1Arr)*w_dist(delta, 1e-4, wRange), label = r'$\epsilon_w = 10^{-4}$', \
         color='darkgreen', lw=1.75)

" tomography "
plt.plot(wRange, [sample_need_tomography(delta, 1e-2)]*len(wRange), color='tab:blue', \
         ls= '--', lw=1.75)
plt.plot(wRange, [sample_need_tomography(delta, 1e-3)]*len(wRange), color='tab:red',  \
         ls= '--', lw=1.75)
plt.plot(wRange, [sample_need_tomography(delta, 1e-4)]*len(wRange), color='darkgreen',\
         ls= '--', lw=1.75)
    
" distillation with white noise "
w_dist_white = np.vectorize(sample_need_dist_white)

plt.plot(wRange, (2. - p1ArrWhite)*w_dist_white(delta, 1e-2, wRange, S), color='tab:blue', \
         ls = (0, (12,5,1,5)),lw=1.75)
plt.plot(wRange, (2. - p1ArrWhite)*w_dist_white(delta, 1e-3, wRange, S), color='tab:red',  \
         ls = (0, (12,5,1,5)),lw=1.75)
plt.plot(wRange, (2. - p1ArrWhite)*w_dist_white(delta, 1e-4, wRange, S), color='darkgreen',\
         ls = (0, (12,5,1,5)),lw=1.75)

    
plt.xlim([0,2./3.])
plt.ylim([1e5, 2e11])

plt.xlabel(r'$w$', fontsize = 36, labelpad=-3)
plt.ylabel(r'$N_{\mathrm{consumed}}$', fontsize = 36, labelpad=-3)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.yscale('log')

plt.tick_params(axis='x', which='major', length=8)
plt.tick_params(axis='y', which='major', length=8)
plt.tick_params(axis='y', which='minor', length=4.5)

plt.legend(loc='upper left', fontsize = 19, ncol=3, columnspacing=1.,\
           frameon=False)

plt.tight_layout()     

plt.savefig('wernerSamples_Delta1e-2.png', format = 'png', dpi = 300)

plt.show()


