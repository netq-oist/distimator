"""
DISTI-MATOR CODE: WERNER STATE ESTIMATION

This .py module contains the code for the Werner state estimation, relying
on vectorizedBell.py and distillationExperiment.py modules. The inversion 
strategy in this code uses a bisection search algorithm.

Consult the supplementary material for the definitions in this code. 

"""

import numpy as np
import distillationExperiment as dx

""""""""""""

"BISECTION METHOD I"

""""""""""""

# Define the inversion strategy for the Werner parameter
# when precision on w is fixed.
# We consider the FIRST distillation protocol for this estimation

def invertWernerParamI(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                       tArray,rateExp,epsw):

  # initialize the w boundary values
  wL = 0.
  wR = 2./3.

  # count iterations
  i = 0
  # set bisection tolerance to be epsw^3
  epsbis = epsw**3.
  # convergence test, i must be at most nLim
  nLim = np.ceil(np.log2((wR - wL)/epsbis))

  ## BISECTION SEARCH
  while i <= nLim:
    # Take the midpoint wM
    wM = (wR + wL)/2.

    # In Bell vector forms
    wMvec = np.array([1.-0.75*wM,0.25*wM,0.25*wM,0.25*wM])

    # Calculate the corresponding rates for wM
    rateM = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                         wMvec,wMvec,tArray,False)

    if np.isclose(rateM, rateExp) or (wR - wL)/2 <= epsbis:
      break

    ## Change the boundaries
    ## Use the fact that over the range w = [0,1),
    ## rate is monotonically decreasing
    if (rateM - rateExp) >= 0.:
      wL = wM
    else:
      wR = wM

    i += 1
    
  return wM
                         
