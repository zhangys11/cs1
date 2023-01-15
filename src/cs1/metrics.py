import numpy as np
import math
import pylab
import matplotlib
import matplotlib.pyplot as plt
import os
import sys


if __package__:
    from . import PSI_LONGNAMES, PSI_MC, PSI_NAMES, GetSensingMatrix
else:
    DIR = os.path.dirname(__file__) # cs1 dir
    if DIR not in sys.path:
        sys.path.insert(0,DIR)
    from __init__ import PSI_LONGNAMES, PSI_MC, PSI_NAMES, GetSensingMatrix

def mutual_coherence(A,B):
    '''
    CS requires PSI and PHI/OMEGA to be incoherent.
    '''

    assert (A.shape == B.shape)
    assert (A.shape[0] == A.shape[1])

    n = A.shape[0]
    p = 0
    
    for i in range(n):
        for j in range(n):
            a = A[:,i].T
            b = B[:,j]            
            tmp = a@b
            # print(tmp)
            if (p < tmp):
                p = tmp
                
    return math.sqrt(n)*p

def sparsity(x, flavor = 'gini'):
    '''
    There lacks a concensus on the sparsity measure.
    This function provides two flaovrs: Lp-norm(0<=p<1) and the Gini index as such a measure.
    
    Parameter
    ---------
    x : input data, 1d array
    flavor : 'L0_inverse' is (1 - L0 Norm / N). This flavor is sensative to noises.
             'gini_v1' and 'gini_v2' will return very close results.
    '''

    if flavor == 'L0_inverse':

        return (len(x) - np.linalg.norm(x, 0))/len(x)

    elif flavor == 'gini_v1' or flavor == 'gini':

        # (Warning: This is a concise implementation, but it is O(n**2)
        # in time and memory, where n = len(x).  *Don't* pass in huge samples!)

        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g

    elif flavor == 'gini_v2':
        
        # Calculate the Gini coefficient of a numpy array. 
        # based on bottom eq:
        # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
        # from:
        # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        # All values are treated equally, arrays must be 1d
        array = np.array(x, dtype = 'float16').flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1,array.shape[0]+1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    
def compare_sparsity_flavors():

    s1 = []
    s2 = []
    s3 = []

    for i in range(100):
        v =  [0]*i + [1]*(100-i) 
        s1.append(sparsity (v, flavor = 'L0_inverse'))
        s2.append(sparsity (v, flavor = 'gini_v1'))
        s3.append(sparsity (v, flavor = 'gini_v2'))


    matplotlib.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24,7))

    fig.suptitle("Calculate sparsity measures on a 100-element array. \nThe zero elements are increased from 0 to 100%")

    ax[0].plot(s1, color = 'gray')
    ax[0].set_ylabel('L0 reverse')
    ax[0].set_xlabel('zero percentage')

    ax[1].plot(s2, color = 'gray')
    ax[1].set_ylabel('GINI v1')
    ax[1].set_xlabel('zero percentage')

    ax[2].plot(s3, color = 'gray')
    ax[2].set_ylabel('GINI v2')
    ax[2].set_xlabel('zero percentage')

    fig.tight_layout()
    plt.show()

    matplotlib.rcParams.update({'font.size': 12})


def analyze_sparsity (x, PSIs, compute_mc = False):
    '''
    Parameters
    ----------
    x : a 1D signal. Will convert to shape (1,n)
    PSIs : a dict of PSI matrices
    compute_mc : whether to computer the mutual coherence
    '''

    # x = x.reshape((-1,1))
    # print("reshape x to ", x.shape)

    plt.figure(figsize=(24,48))
    rows = len(PSIs)
    matplotlib.rcParams.update({'font.size': 18})

    for idx, key in enumerate(PSIs):
        
        PSI = PSIs[key]        
        xx = np.copy(x)
        
        # pad x with zero if necessary, e.g., the HWT case
        if (len(x) < PSI.shape[0]):
            xx = np.zeros(PSI.shape[0])
            xx[:len(x)] = x        
        
        # PSI_H = PSI.conj().T
        # PSI_INV = np.linalg.pinv(PSI)
        #print(PSI_H)
        #print(PSI_INV)

        # theoretically, PSI_H == PSI_INV
        z = PSI @ xx
        # print(z)
        MAX = abs(max(z, key=abs))
        # print(MAX)
        thresholds = np.array(range(100)) / 5000
        
        rs =[]
        for threshold in thresholds:
            rs.append((np.abs(np.array(z)) <= threshold * MAX).sum() / len(z))
            
        auc = 0
        for i in range(1000):
            auc += (np.abs(np.array(z)) <= (i+1)/1000 * MAX).sum() / len(z)        
        auc = auc/1000
        
        r = (np.abs(np.array(z)) <= 0.02 * MAX).sum() / len(z)  # use 0.02 MAX ABS as threshold
        
        plt.subplot(rows,2,2*idx+1)
        if np.any(np.iscomplex(z)):
            plt.plot(np.abs(z), color='gray')
        else:
            plt.plot(z, color='gray')
            
        title = key
        if key in PSI_NAMES:
            psi_idx = PSI_NAMES.index(key)
            title = PSI_LONGNAMES[ psi_idx ] + ' (' + key + '), MC = ' + PSI_MC[ psi_idx ]
            
            if compute_mc:
                n = PSIs[key].shape[0]
                OMEGA = np.identity(n)
                print("Mutual Coherence (" + key, ", OMEGA) : ", mutual_coherence(PSIs[key], OMEGA))
            
        plt.title( title )
        # plt.axis('off')
        
        plt.subplot(rows,2,2*idx+2)
        plt.scatter(thresholds, rs, color='gray')
        plt.title('AUC = ' + str(round(auc,3)) + ', $r_{0.02max}$ = ' + str(round(r,3)) \
            + ', gini = ' + str(round(sparsity(z),3)))
        pylab.xlim([0,0.02])
        pylab.ylim([0,1.0])
        plt.xticks(np.arange(0.0, 0.021, 0.005))

    plt.show()    
    matplotlib.rcParams.update({'font.size': 12})

def Univariate_KLD(p, q):

    # p is target distribution
    return np.log(q[1] / p[1]) + (p[1] ** 2 + (p[0] - q[0]) ** 2) / (2 * q[1] ** 2) - 0.5


def Multivarate_KLD(p, q):

    a = np.log(np.linalg.det(q[1])/np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)

##  https://github.com/zhangys11/pyDRMetrics/blob/main/src/pyDRMetrics/calculate_recon_error.py 

def recon_MSE(X,Xr):
    '''
    Reconstruction relateive MSE
    '''
    assert X.shape == Xr.shape
    return np.linalg.norm(X-Xr, ord='fro')**2/(X.shape[0]*X.shape[1])

def calculate_recon_error(X, Xr):
    '''
    Recontruction errors

    Return
    ------
    mse : mean squared error between X and Xr
    ams : mean square of X
    mse/ams : relative mean squared error
    '''
    assert X.shape == Xr.shape

    mse = recon_MSE(X, Xr) # mean square error
    ams = recon_MSE(X, np.zeros(X.shape)) # mean square of original data matrix

    return mse, ams, mse/ams

def recon_rMSE(X, Xr):
    '''
    relative mean squared error
    '''
    return recon_MSE(X, Xr) / recon_MSE(X, np.zeros(X.shape))

def SNR(X, Xr):
    '''
    SNR (signal-Noise Ratio in dB). dB = 10*log10
    '''
    return 10 * np.log10(1.0 / recon_rMSE(X, Xr) )