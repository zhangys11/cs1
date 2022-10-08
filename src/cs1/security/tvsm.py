'''
TVSM related functions
'''

import time
from math import factorial, log10
import random
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ..cs import MeasurementMatrix, Recovery

def Time2Seed(t = None):

    '''
    Parameter
    ---------
    t : a timestamp. If unspecified, will use "now".

    Return
    ------
    Seconds since epoch. Unix and POSIX measure time as the number of seconds that have passed since 1 January 1970 00:00:00 UT, a point in time known as the Unix epoch. The NT time epoch on Windows NT and later refers to the Windows NT system time in (10^-7)s intervals from 0h 1 January 1601.
    '''
    
    if t is None:
        t = time.time()
    return int(round(t)) # Return the current time in seconds since the Epoch.

def Seed2Time(s):
    return time.ctime(s)

def brute_force_total_guesses(N = 3000, K = 100, verbose = True):
    '''
    we simulate a security senario: a hacker intercepts our transmmitted signal xs (len(xs) = N). 
    He/she tries to guess the sensing matrix (samples = k) and reconstruct the signal with the brute-force method.
    This function returns the search space, i.e., total guesses needed to iterate all the combinations.

    Parameters
    ----------
    N : original signal length
    K : sampled points

    Return
    ------
    total guesses in base 10.
    '''

    print('N = ', N, ', K = ', K)
    LOG10C = log10(factorial(N)) - log10(factorial(N - K)) 
    print('A(N,K) in log10 = ', LOG10C ) 
    print('Total years needed to iterate all combinations (in log10) = ', LOG10C - log10(3600 * 24 * 365.25))

    return LOG10C

def brute_force_guess_plot(N = 3000):
    '''
    Plot how the search space changes with data dim
    '''

    plt.figure(figsize=(16,12))
    for k in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:        
        A = []
        for n in N:
            K = round(n * k)
            a = brute_force_total_guesses(n, K, verbose = False) # log10(factorial(n)) - log10(factorial(n - K))
            A.append(a)
        plt.plot(N, A, label = 'k = ' + str(k) )

    plt.ylabel('search space ($ N_\Phi $) in log 10')
    plt.xlabel('signal dimensionality (n)')
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()
    plt.show()
    

def brute_force_guess(xs, N, t = 'DCT', ITER = 100000):
    '''
    Paramters
    ---------
    xs : sampled data
    N : original signal dim
    t : PSI. default is DCT
    ITER : randomly pick ITER sensing matrix to test the reconstruction
    '''
    

    BEST_IDX = []
    MAX_R = 0
    MAX_AUC = 0
    BEST_XR = None
    BEST_Z = None
    
    K = len(xs)

    for i in tqdm( range(ITER) ):
        idx = random.sample(range(N), K)
        A = MeasurementMatrix(N, idx, t)
        xr, z = Recovery (A, xs, t = t, silent = True)        
        
        MAX = abs(max(z, key=abs))
        thresholds = np.array(range(100)) / 5000
        
        auc = 0
        for i in range(1000):
            auc += (np.abs(np.array(z)) <= (i+1)/1000 * MAX).sum() / len(z)        
        auc = auc/1000
        
        # calculate sparsity
        r = (np.abs(np.array(z)) <= 0.02 * MAX).sum() / len(z)  # use 0.01 MAX ABS as threshold  

        if (MAX_R < r):
            MAX_R = r
            MAX_AUC = auc
            BEST_IDX = idx
            BEST_XR = xr
            BEST_Z = z
            
    # plot the best result after ITER iterations

    print('Best AUC = ', round(MAX_AUC, 3), '\nsparsity = ', round(MAX_R, 3))

    matplotlib.rcParams.update({'font.size': 18})

    plt.figure(figsize=(12,3))
    plt.plot(BEST_Z, color='gray')
    # plt.title('reconstructed signal in the latent space (z)')
    plt.xticks([])
    plt.yticks([])
    plt.title('z best guess (with highest sparsity)')
    plt.show()

    plt.figure(figsize=(12,3))
    plt.plot(BEST_XR, color='gray')
    # plt.title('reconstructed signal (xr)')
    plt.xticks([])
    plt.yticks([])
    plt.title('xr best guess')
    plt.show()

    return BEST_Z, BEST_XR