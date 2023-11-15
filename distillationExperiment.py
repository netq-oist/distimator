"""
DISTI-MATOR CODE: DISTILLATION SIMULATION

This .py module contains definitions 
that calculate for the success probabilities of each distillation protocol, 
and to simulate experiments via an accept/reject criterion 

Consult the supplementary material for the definitions in this code. 

"""

import numpy as np
import vectorizedBell as vb

""""""""""""

"DISTILLATION PROTOCOLS"

""""""""""""

"PROTOCOL 1"

def protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                 pvec,qvec,tArray,expt):
  # Total number of samples
  Nsamples = np.shape(tArray)[0]

  pvecArr = np.vstack([pvec]*Nsamples)
  qvecArr = np.vstack([qvec]*Nsamples)

  zetaArrA = 0.5*(1. - np.exp(-tArray/TdphA))
  ldaArrA = 1. - np.exp(-tArray/TdpoA)

  zetaArrB = 0.5*(1. - np.exp(-tArray/TdphB))
  ldaArrB = 1. - np.exp(-tArray/TdpoB)

  ### CALCULATE SUCCESS PROBABILITY
  # Noisy preparation
  pnewArr = vb.noisyPreparation(pvecArr,ldaArrA,ldaArrB,zetaArrA,zetaArrB)

  # Noisy CNOT
  fArr = vb.noisyCNOT(pnewArr, qvecArr, yA, yB)

  # Noisy measurement
  rateArr = vb.noisyMeasurementZtarget(fArr, etaZA, etaZB)
  # array of expected rates as a function of t

  if expt == True:
    ### Simulated experiment 
    u = np.random.uniform(0.,1., Nsamples)

    # accept or reject
    booleArr = np.where(u < rateArr, 1., 0.)

    # return empirical average
    return np.sum(booleArr)/Nsamples

  else:
    # return expected success probability
    return np.sum(rateArr)/Nsamples

"PROTOCOL 2"

def protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                 pvec,qvec,tArray,expt):
  # Total number of samples
  Nsamples = np.shape(tArray)[0]

  pvecArr = np.vstack([pvec]*Nsamples)
  qvecArr = np.vstack([qvec]*Nsamples)

  zetaArrA = 0.5*(1. - np.exp(-tArray/TdphA))
  ldaArrA = 1. - np.exp(-tArray/TdpoA)

  zetaArrB = 0.5*(1. - np.exp(-tArray/TdphB))
  ldaArrB = 1. - np.exp(-tArray/TdpoB)

  ### CALCULATE SUCCESS PROBABILITY
  # Noisy preparation
  pnewArr = vb.noisyPreparation(pvecArr,ldaArrA,ldaArrB,zetaArrA,zetaArrB)

  # Noisy CNOT
  fArr = vb.noisyCNOT(pnewArr, qvecArr, yA, yB)

  # Noisy measurement
  rateArr = vb.noisyMeasurementXsource(fArr, etaXA, etaXB)
  # array of expected rates as a function of t

  if expt == True:
    ### Simulated experiment 
    u = np.random.uniform(0.,1., Nsamples)

    # accept or reject
    booleArr = np.where(u < rateArr, 1., 0.)

    # return empirical average
    return np.sum(booleArr)/Nsamples

  else:
    # return expected success probability
    return np.sum(rateArr)/Nsamples

"PROTOCOL 3"

def protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                 pvec,qvec,tArray,expt):
  # Total number of samples
  Nsamples = np.shape(tArray)[0]

  pvecArr = np.vstack([pvec]*Nsamples)
  qvecArr = np.vstack([qvec]*Nsamples)

  zetaArrA = 0.5*(1. - np.exp(-tArray/TdphA))
  ldaArrA = 1. - np.exp(-tArray/TdpoA)

  zetaArrB = 0.5*(1. - np.exp(-tArray/TdphB))
  ldaArrB = 1. - np.exp(-tArray/TdpoB)

  ### CALCULATE SUCCESS PROBABILITY
  # Noisy preparation
  pnewArr = vb.noisyPreparation(pvecArr,ldaArrA,ldaArrB,zetaArrA,zetaArrB)

  # Noisy local rotations
  pnewArr = vb.noisyRotationX(pnewArr, mA, mB)
  qnewArr = vb.noisyRotationX(qvecArr, mA, mB)

  # Noisy CNOT
  fArr = vb.noisyCNOT(pnewArr, qnewArr, yA, yB)

  # Noisy measurement
  rateArr = vb.noisyMeasurementZtarget(fArr, etaZA, etaZB)
  # array of expected rates as a function of t

  if expt == True:
    ### Simulated experiment 
    u = np.random.uniform(0.,1., Nsamples)

    # accept or reject
    booleArr = np.where(u < rateArr, 1., 0.)

    # return empirical average
    return np.sum(booleArr)/Nsamples

  else:
    # return expected success probability
    return np.sum(rateArr)/Nsamples
