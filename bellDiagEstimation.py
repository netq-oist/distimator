"""
DISTI-MATOR CODE: BELL-DIAGONAL STATE ESTIMATION

This .py module contains the code for the Bell-diagonal state estimation, 
relying on vectorizedBell.py and distillationExperiment.py modules. 
The inversion strategy in this code uses a bisection search algorithm.

Consult the supplementary material for the definitions in this code. 

"""

import numpy as np
import distillationExperiment as dx

""""""""""""

"BISECTION METHOD I"

""""""""""""

# Define the inversion strategy for the Bell-diagonal state
# when precisions on x_i's are initially determined,
# described by an array of epsxArr
# We consider the THREE distillation protocols

def invertBellProtocolI(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                        etaXA,etaXB,tArr1,tArr2,tArr3,pExpArr,epsxArr):
    
    "FIRST distillation protocol"
    # Initialize the x1 values
    x1L = 1./2.
    x1R = 1.
    
    # count iterations
    x1i = 0
    # set bisection tolerance to be 10^-2 x epsx
    epsbis1 = (1e-2)*epsxArr[0]
    # convergence test, total step must be at most nLim1
    nLim1 = np.ceil(np.log2((x1R - x1L)/epsbis1))
    
    ## BISECTION SEARCH
    while x1i <= nLim1:
        # Take the midpoint x1M
        x1M = (x1R + x1L)/2.
      
        ## Since the success probability in the first distillation
        ## protocol only depends on q1 and q2, we can simply set 
        ## q2 = 0  and q4 = 0, giving q1 = x1 and q3 = 1 - x2.
        ## This follows an additional assumption that 
        ## q1 > q2 >= q3 >= q4 (up to some permutation of q2,q3,q4)
        
        # In the Bell vector form
        x1Mvec = np.array([x1M, 0., 1.-x1M, 0.])
        
        # Calculate the corresponding success probability for x1Mvec
        p1M = dx.protocolExp1(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaZA,etaZB,\
                              x1Mvec,x1Mvec,tArr1,False)
            
        if np.isclose(p1M, pExpArr[0]) or (x1R - x1L)/2 <= epsbis1:
            break
        
        ## Change the boundaries
        ## Use the fact that over the range x=(1/2,1),
        ## the success probability is monotonically INCREASING
        if (p1M - pExpArr[0]) >= 0.:
            x1R = x1M
        else:
            x1L = x1M
            
        x1i += 1
        
    "SECOND distillation protocol"
    # Initialize the x2 values
    x2L = 1./2.
    x2R = 1.
    
    # count iterations
    x2i = 0
    # set bisection tolerance to be 10^-2 x epsx
    epsbis2 = (1e-2)*epsxArr[1]
    # convergence test, total steps must be at most nLim2
    nLim2 = np.ceil(np.log2((x2R - x2L)/epsbis2))
    
    ## BISECTION SEARCH
    while x2i <= nLim2:
        # Take the midpoint x2M
        x2M = (x2R + x2L)/2.
    
        ## Since the success probability in the second distillation
        ## protocol only depends on q1 and q3, we can simply set 
        ## q3 = 0 and q4 = 0, giving q1 = x2 and q2 = 1 - x2.
        ## This follows an additional assumption that 
        ## q1 > q2 >= q3 >= q4
        
        # In the Bell vector form
        x2Mvec = np.array([x2M, 1.-x2M, 0., 0.])
        
        # Calculate the corresponding success probability for x2Mvec
        p2M = dx.protocolExp2(TdpoA,TdphA,TdpoB,TdphB,yA,yB,etaXA,etaXB,\
                              x2Mvec,x2Mvec,tArr2,False)
            
        if np.isclose(p2M, pExpArr[1]) or (x2R - x2L)/2 <= epsbis2:
            break
        
        ## Change the boundaries
        ## Use the fact that over the range x=(1/2,1),
        ## the success probability is monotonically INCREASING
        if (p2M - pExpArr[1]) >= 0.:
            x2R = x2M
        else:
            x2L = x2M
            
        x2i += 1
        
    "THIRD distillation protocol"
    # Initialize the x3 values
    x3L = 1./2.
    x3R = 1.
    
    # count iterations
    x3i = 0
    # set bisection tolerance to be 10^-2 x epsx
    epsbis3 = (1e-2)*epsxArr[2]
    # convergence test, total steps must be at most nLim3
    nLim3 = np.ceil(np.log2((x3R - x3L)/epsbis3))
    
    ## BISECTION SEARCH
    while x3i <= nLim3:
        # Take the midpoint x2M
        x3M = (x3R + x3L)/2.
        
        ## Since the success probability in the third distillation
        ## protocol only depends on q1 and q4, we can simply set 
        ## q4 = 0 and q3 = 0, giving q1 = x3 and q2 = 1 - x3.
        ## This follows an additional assumption that 
        ## q1 > q2 >= q3 >= q4
        
        # In the Bell vector form
        x3Mvec = np.array([x3M, 1.-x3M, 0., 0.])
        
        # Calculate the corresponding success probability for x2Mvec
        p3M = dx.protocolExp3(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                              x3Mvec,x3Mvec,tArr3,False)
            
        if np.isclose(p3M, pExpArr[2]) or (x3R - x3L)/2 <= epsbis3:
            break
        
        ## Change the boundaries
        ## Use the fact that over the range x=(1/2,1),
        ## the success probability is monotonically INCREASING
        if (p3M - pExpArr[2]) >= 0.:
            x3R = x3M
        else:
            x3L = x3M
            
        x3i += 1
    
    ## Return the x values 
    return np.array([x1M,x2M,x3M])

""""""""""""

"x vector to q vector"

""""""""""""

def convertToQ(xArr):
    "Calculate for the q vector"
    q1 = 0.5*(-1. + xArr[0] + xArr[1] + xArr[2])
    q2 = 0.5*( 1. + xArr[0] - xArr[1] - xArr[2])
    q3 = 0.5*( 1. - xArr[0] + xArr[1] - xArr[2])
    q4 = 0.5*( 1. - xArr[0] - xArr[1] + xArr[2])

    return np.array([q1,q2,q3,q4])

""""""""""""

"BISECTION METHOD II"

""""""""""""

# Define the inversion strategy for the Bell-diagonal state
# when precisions on p^i's are initially determined,
# described by an array of epspArr
# We consider the THREE distillation protocols

def invertBellProtocolII(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,
                         etaXA,etaXB,tArr1,tArr2,tArr3,pExpArr,epspArr):

    # Search precision:
    # Using a bisection search when epsx is unknown
    # Typically, epsp = order(epsx). 
    # So, as a rule of thumb, we take epsx_bis = 10**-2 epsp
    # as precision of the search algorithm
    epsxArr = epspArr*10.**(-2.)
    
    xArr = invertBellProtocolI(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                               etaXA,etaXB,tArr1,tArr2,tArr3,pExpArr,epsxArr)
    
    xArrL = invertBellProtocolI(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                   etaXA,etaXB,tArr1,tArr2,tArr3,\
                                       pExpArr-epspArr,epsxArr)
        
    xArrR = invertBellProtocolI(TdpoA,TdphA,TdpoB,TdphB,mA,mB,yA,yB,etaZA,etaZB,\
                                   etaXA,etaXB,tArr1,tArr2,tArr3,\
                                       pExpArr+epspArr,epsxArr)    
        
    ## Return the results
    return xArr, xArrL, xArrR
    
        
    
    
    
    
    
