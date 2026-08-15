
"""
@author: jcasapao

"""

import stim
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

plt.style.use('classic')

class QuantumErrorEstimator:
    def __init__(self, qx, qy, qz):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.q0 = 1.0 - qx - qy - qz
        
        ### Pre-compute the sympy expressions to avoid heavy looping during sampling
        self.symX, self.symY, self.symZ = sy.symbols('x y z', real=True)
        self.jointExpr = self._precomputeJointExpr()
        
    def _precomputeJointExpr(self):
        """Builds and simplifies the joint probability sympy expression once."""
        def getVar(n):
            if n == 0: return 1 - self.symX - self.symY - self.symZ
            if n == 1: return self.symZ
            if n == 2: return self.symX
            if n == 3: return self.symY

        expr = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if ((i ^ j ^ k) & 2) == 0 and ((j ^ k) & 1) == 0:
                        expr += getVar(i) * getVar(j) * getVar(k)
        
        return sy.simplify(expr)

    def getSamples(self, numShots):
        circuitString = f"""
        H 0 1 2
        CNOT 0 3 1 4 2 5
        PAULI_CHANNEL_1({self.qx}, {self.qy}, {self.qz}) 0 1 2
        CNOT 0 1 3 4
        CNOT 2 1 5 4
        MZ 1 4
        MX 2 5
        """
        circuit = stim.Circuit(circuitString)
        sampler = circuit.compile_sampler()
        result = sampler.sample(shots=numShots)

        pArr = np.einsum('ij->ji', result)
        zzArr = ~(pArr[0] ^ pArr[1])
        xxArr = ~(pArr[2] ^ pArr[3])
        jointSuccessArr = zzArr * xxArr

        return zzArr, xxArr, jointSuccessArr

    def estimateParams(self, zzProb, xxProb, jointProb):
        """Solves the system of equations for the given empirical probabilities."""
        eq1 = sy.Eq(zzProb, (1 - self.symX - self.symY)**3 + 3*(1 - self.symX - self.symY)*(self.symX + self.symY)**2)
        eq2 = sy.Eq(xxProb, (1 - self.symY - self.symZ)**2 + (self.symY + self.symZ)**2)
        eq3 = sy.Eq(jointProb, self.jointExpr)

        sol = sy.nsolve((eq1, eq2, eq3), (self.symX, self.symY, self.symZ), (0.0, 0.0, 0.0))
        return np.array(sol).astype(np.float64).flatten()

    def plotInstanceTracking(self, maxShots=10_000_000, trackStep=1_000, plotFilename='DS-numerical-single-instance.pdf', dataFilename='trackingData.txt'):
        """Tracks the convergence of parameter estimations over an increasing number of shots."""
        zzArr, xxArr, jointArr = self.getSamples(maxShots)
        
        zzCum = np.cumsum(zzArr)
        xxCum = np.cumsum(xxArr)
        jointCum = np.cumsum(jointArr)
        
        numSamples = np.arange(trackStep, maxShots + 1, trackStep)
        
        estimatedX, estimatedY, estimatedZ = [], [], []
        
        for n in numSamples:
            zzP = zzCum[n-1] / n
            xxP = xxCum[n-1] / n
            jointP = jointCum[n-1] / n
            
            xEst, yEst, zEst = self.estimateParams(zzP, xxP, jointP)
            estimatedX.append(xEst)
            estimatedY.append(yEst)
            estimatedZ.append(zEst)
            
        estX = np.array(estimatedX)
        estY = np.array(estimatedY)
        estZ = np.array(estimatedZ)
        fidelities = 1.0 - estX - estY - estZ
        
        ### Write tracking data to txt
        with open(dataFilename, 'w') as fileOut:
            fileOut.write("NumShots Fidelity q2Est q3Est q4Est\n")
            for i, n in enumerate(numSamples):
                fileOut.write(f"{n} {fidelities[i]:.6f} {estZ[i]:.6f} {estX[i]:.6f} {estY[i]:.6f}\n")

        ### Format & Plot (matches experiment_plot_new layout)
        plt.rcParams.update({
            'font.size': 36,
            'font.family': 'serif',
            'text.usetex': True
        })
        
        fig, axs = plt.subplots(4, 1, figsize=(15, 40))
        
        axs[0].plot(numSamples, fidelities, label=r'$\hat{q}_1$', lw=3)
        axs[0].plot(numSamples, [self.q0]*len(numSamples), '--', label=r'$q_1$', lw=3)
        axs[0].legend()
        axs[0].set_ylim(bottom=self.q0 - 0.01, top=self.q0 + 0.01)
        
        axs[1].plot(numSamples, estZ, label=r'$\hat{q}_2$', lw=3)
        axs[1].plot(numSamples, [self.qz]*len(numSamples), '--', label=r'$q_2$', lw=3)
        axs[1].legend()
        axs[1].set_ylim(bottom=self.qz - 0.01, top=self.qz + 0.01)

        axs[2].plot(numSamples, estX, label=r'$\hat{q}_3$', lw=3)
        axs[2].plot(numSamples, [self.qx]*len(numSamples), '--', label=r'$q_3$', lw=3)
        axs[2].legend()
        axs[2].set_ylim(bottom=self.qx - 0.01, top=self.qx + 0.01)

        axs[3].plot(numSamples, estY, label=r'$\hat{q}_4$', lw=3)
        axs[3].plot(numSamples, [self.qy]*len(numSamples), '--', label=r'$q_4$', lw=3)
        axs[3].legend()
        axs[3].set_ylim(bottom=self.qy - 0.01, top=self.qy + 0.01)
        axs[3].set_xlabel(r'$N$', fontsize=36, labelpad=-1)
        
        fig.tight_layout()
        plt.savefig(plotFilename, format='pdf', dpi=300)
        plt.show()

    def plotHistogram(self, maxShots=10_000_000, repeats=1_000, plotFilename='DS-numerical-histogram.pdf', dataFilename='histogramData.txt'):
        """Plots the trace distance frequencies over numerous experimental trials."""
        qArrDiff = np.zeros((repeats, 4))
        
        for j in range(repeats):
            zzArr, xxArr, jointArr = self.getSamples(maxShots)
            
            # Since we only care about the maxShots point, we don't need cumsum here
            zzP = np.sum(zzArr) / maxShots
            xxP = np.sum(xxArr) / maxShots
            jointP = np.sum(jointArr) / maxShots
            
            estX, estY, estZ = self.estimateParams(zzP, xxP, jointP)
            estQ0 = 1.0 - estX - estY - estZ
            
            qArrDiff[j, 0] = estQ0 - self.q0
            qArrDiff[j, 1] = estX - self.qx
            qArrDiff[j, 2] = estY - self.qy
            qArrDiff[j, 3] = estZ - self.qz
            
        # Trace distance
        trDist = 0.5 * np.sum(np.abs(qArrDiff), axis=1)
        
        # Calculate statistics
        counts, bins = np.histogram(trDist, bins='fd')
        mids = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(mids, weights=counts)
        var = np.average((mids - mean)**2, weights=counts)
        std = np.sqrt(var)
        
        ### Write statistics and raw distances to txt
        with open(dataFilename, 'w') as fileOut:
            fileOut.write(f"Mean: {mean:.8f}\nStdDev: {std:.8f}\n")
            fileOut.write("TraceDistances\n")
            for dist in trDist:
                fileOut.write(f"{dist:.6f}\n")
                
        print(f"[{plotFilename}] Mean: {mean:.6f} | Std: {std:.6f}")
        
        ### Format & Plot
        plt.rcParams.update({'font.size': 18, 'font.family': 'serif', 'text.usetex': True})
        fig, ax = plt.subplots()
        
        ax.hist(bins[:-1], bins, weights=counts)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_xlabel(r'$D(\hat{\rho}(\hat{\mathbf{q}}),\overline{\rho}(\mathbf{q}))$', fontsize=32)
        ax.set_ylabel(r'$\mathrm{Frequency}\,(10^4\,\mathrm{trials})$', fontsize=32)
        
        fig.tight_layout()
        plt.savefig(plotFilename, format='pdf', dpi=300)
        plt.show()

#%%

########################################
# Running the code
########################################

if __name__ == "__main__":
    # Standard Case (qx=0.05, qy=0.1, qz=0.15)
    standardEstimator = QuantumErrorEstimator(qx=0.05, qy=0.1, qz=0.15)
    standardEstimator.plotInstanceTracking(maxShots=50_000, trackStep=500, plotFilename='DS-numerical-single-instance.pdf', dataFilename='standardTracking.txt')
    
    # Lossy Histogram
    lossyEstimator = QuantumErrorEstimator(qx=0.05, qy=0.18, qz=0.15)
    lossyEstimator.plotHistogram(maxShots=10_000, repeats=10_000, plotFilename='DS-numerical-histogram-lossy-0.62-0.15-0.05-0.18.pdf', dataFilename='lossyHistogram.txt')

    # High Fidelity Histogram
    highFidEstimator = QuantumErrorEstimator(qx=0.05, qy=0.05, qz=0.02)
    highFidEstimator.plotHistogram(maxShots=10_000, repeats=10_000, plotFilename='DS-numerical-histogram-high0.88-0.02-0.05-0.05.pdf', dataFilename='highFidHistogram.txt')