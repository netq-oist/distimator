"""
DISTI-MATOR CODE: BELL VECTORIZATION 

This .py module contains a Bell vectorization of the states and operations 
involved in the three distillation rounds. 

Consult the supplementary material for the definitions in this code. 

"""

import numpy as np

""""""""""""

"BELL VECTORIZATION"

""""""""""""

"Noise in the preparation stage"

def noisyPreparation(pvec,ldaA,ldaB,zetaA,zetaB):
  # depolarizing B
  pnew = 0.25*ldaB[...,None]*np.ones(np.shape(pvec)) \
         + (1.-ldaB[...,None])*pvec

  # dephasing B
  idx = np.array([1,0,3,2])
  pnew = (1.-zetaB[...,None])*pnew + zetaB[...,None]*pnew[:,idx]

  # depolarizing A
  pnew = 0.25*ldaA[...,None]*np.ones(np.shape(pnew)) \
         + (1.-ldaA[...,None])*pnew

  # dephasing A
  pnew = (1.-zetaA[...,None])*pnew + zetaA[...,None]*pnew[:,idx]

  return pnew

"Noisy rotations"

def noisyRotationX(pvec, mA, mB):
  idx = np.array([0,3,2,1])
  pnew = (1.-mA)*(1.-mB)*pvec[:,idx] \
          + 0.25*(mA + mB - mA*mB)*np.ones(np.shape(pvec))

  return pnew

"Noisy CNOT"

def noisyCNOT(pvec, qvec, yA, yB):
  pidx = np.array([0,1,0,1,1,0,1,0,2,3,2,3,3,2,3,2])
  qidx = np.array([0,1,2,3,0,1,2,3,2,3,0,1,2,3,0,1])

  fvec = (1.-yA)*(1.-yB)*pvec[:,pidx]*qvec[:,qidx] \
       + (1./16.)*(yA + yB - yA*yB)*np.ones((np.shape(pvec)[0],16))

  return fvec

"Noisy measurement"

## Here, we are multiplying the rates by 1/2 since we
## only want the POVMs M00 and M++

def noisyMeasurementZtarget(fvec, etaZA, etaZB):
  fidx = np.array([0,1,4,5,8,9,12,13])
  sidx = np.array([2,3,6,7,10,11,14,15])

  ## success probability
  rate = 0.5*(1. - etaZB - etaZA*(1. - 2.*etaZB))*np.sum(fvec[:,fidx], axis=1) \
                + (etaZB + etaZA*(1. - 2.*etaZB))*np.sum(fvec[:,sidx], axis=1)

  return rate

def noisyMeasurementXsource(fvec, etaXA, etaXB):
  fidx = np.array([0,1,2,3,8,9,10,11])
  sidx = np.array([4,5,6,7,12,13,14,15])

  ## success probability
  rate = 0.5*(1. - etaXB - etaXA*(1. - 2.*etaXB))*np.sum(fvec[:,fidx], axis=1) \
                + (etaXB + etaXA*(1. - 2.*etaXB))*np.sum(fvec[:,sidx], axis=1)

  return rate
