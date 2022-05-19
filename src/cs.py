import matplotlib.pyplot as plt
import pylab, matplotlib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
import time
import matplotlib.ticker as mticker
import cv2
import matplotlib.cm as cm
from sklearn.linear_model import Lasso, LassoCV
import pywt
from scipy.stats import ortho_group
import scipy
from scipy import pi
from scipy.linalg import hadamard
import math
from scipy.stats import ortho_group
import pickle
from sklearn.decomposition import PCA
import scipy.signal as signal
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from pyDRMetrics.pyDRMetrics import *

from plotComponents2D import *


import warnings
warnings.filterwarnings("ignore")

#region Performance Test. Compare NNRW with mainstream classifier models.

PSI_NAMES = ['IDM', 'DCT', 'DFT', 'DWT', 'HWT', 'ROM']
PSI_LONGNAMES = ['Identity Matrix', 'Discrete Cosine Transform', 'Discrete Fourier Transform', 
'Discrete Wavelet Transform', 'Hadamard-Walsh Matrix', 'Random Orthogonal Matrix']
PSI_MC = ['$\sqrt{n}$', '$\sqrt{2}$', '1', 'around 0.8 $\sqrt{n}$', '1', '$\sqrt{2 log(n)}$']

def dctmtx(m, n, display = True):    
    '''
    Return an m-by-n DCT sensing matrix
    '''
    mtx = np.zeros((m,n))
    N = n

    mtx[0, :] = 1 * np.sqrt(1/N) 
    for i in range(1, m):
        for j in range(n):
            mtx[i, j] = np.cos(np.pi *i * (2*j+1) / (2*N)) * np.sqrt(2/N)
    
    if display:
        plt.figure()
        plt.imshow(mtx, interpolation='nearest', cmap=cm.Greys_r)
        plt.axis('off')
        plt.title("DCT (" + str(m) + " , " + str(n) + ")")
        plt.show()
    
    return mtx

def dftmtx(N, flavor = 1, display = True):    

    if flavor == 2:
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        w = np.exp( - 2 * np.pi * 1j / N )
        mtx = np.power( w, i * j ) / np.sqrt(N)

    else:
        mtx = np.zeros((N,N), dtype=np.complex)
        w = np.exp(-2 * np.pi * 1j / N) # python uses j as imaginary unit

        for j in range(N):
            for k in range(N):
                mtx[j, k] = np.power(w, k*j) / np.sqrt(N)

    if display:
        f,ax = plt.subplots(1,3,figsize=(4*3,4))

        f.suptitle("DFT Sensing Matrix using Flavor " + str(flavor))

        plt.subplot(1,3,1)
        plt.imshow(abs(mtx), interpolation='nearest', cmap=cm.Greys_r)  
        plt.axis('off')
        ax[0].set_title('abs')
        
        plt.subplot(1,3,2)
        plt.imshow(np.angle(mtx), interpolation='nearest', cmap=cm.Greys_r)
        plt.axis('off')   
        ax[1].set_title('phase')

        plt.subplot(1,3,3)
        E = np.dot(mtx, mtx)
        plt.imshow(abs(E), cmap=cm.Greys_r)
        plt.axis('off')
        ax[2].set_title('DFT @ DFT.T = I')
    
    return mtx

def hwtmtx(n, display = True):
    '''
    Return a HWT matrix. Its dimension is nearest 2^n above N. 
    '''
    
    NN = 2**(n-1).bit_length() # make sure it is a 2**n value
    print('Expanded to ', NN, ', Divide by', 2**(math.log(NN,2)/2))
    
    mtx = hadamard(NN) / (2**(math.log(NN,2)/2))
    
    # Use slogdet when mtx is big, e.g. > 2000
    logdet = np.linalg.slogdet(mtx)
    det = logdet[0] * np.exp(logdet[1])
    print('det(HWT_MTX) =', round(det, 5))
    
    if display:
        plt.figure()
        plt.imshow(mtx, interpolation='nearest', cmap=cm.Greys_r)
        plt.axis('off')
        plt.title("Hadamard-Walsh Matrix (" + str(NN) + " , " + str(NN) + ")")
        plt.show()

    return mtx, NN

def dwtmtx(N, wavelet = 'db3', display = True):
    '''
    Return a DWT matrix. Its dimension is nearest 2*n (even number) above N. 
    '''
    if N % 2 == 1:
        N = N +1
        print('HWT requires even dimensions. Expanded to ', N)

    mtx = np.zeros((N,N))  
    I = np.identity(N)
        
    for i in range(N):
        cA, cD = pywt.dwt(I[i,:], wavelet, pywt.Modes.periodization)
        # print(cA.shape, cD.shape)
        mtx[i,:] = list(cA) + list(cD)
    
    if display:

        plt.figure()
        plt.imshow(mtx, interpolation='nearest', cmap=cm.Greys_r)
        plt.title(wavelet + " Wavelet Matrix (" + str(N) + " , " + str(N) + ")")
        plt.axis('off')
        plt.show()    
        print('det(DWT_MTX) = ', round(scipy.linalg.det(mtx), 5))   

    return mtx, N

def DwtMcCurve():

    Ns = [10, 20, 50, 100, 150, 200, 500, 1000, 2000] # dimensions
    mcs = []
    for N in Ns:
        
        _, OMEGA = GetSensingMatrix(N)
        psi, _ = dwtmtx(N, display = False)
        mc = Mutual_Coherence(psi, OMEGA)
        # print("N :", N, ".  : ", mc)
        mcs.append(mc)

    plt.plot(Ns, mcs)
    plt.title(r'Mutual Coherence (DWT, OMEGA)')
    plt.show()

    plt.plot(Ns, np.sqrt(Ns) * .8)
    plt.title(r'0.8 $\sqrt{n}$')
    plt.show()

def rvsmtx(N, display = True):

    mtx = ortho_group.rvs(N) # uniformly distributed random orthogonal matrix
    
    if display:
        plt.figure()
        plt.imshow(mtx, interpolation='nearest', cmap=cm.Greys_r)
        plt.title('Random Orthogonal Matrix')
        plt.axis('off')
        plt.show()
    
    print('det(RVS_MTX) = ',  round(scipy.linalg.det(mtx), 5))    
    return mtx

def Generate_PSI(n, psi_type = 1):
    '''
    Parameter
    ---------
    psi_type : one of PSI_NAMES or its index
    '''
    if psi_type == PSI_NAMES[0] or psi_type == 0:
        return np.identity(n)

    if psi_type == PSI_NAMES[1] or psi_type == 1:
        return dctmtx(n,n, display = False)

    if psi_type == PSI_NAMES[2] or psi_type == 2:
        return dftmtx(n, display = False)

    if psi_type == PSI_NAMES[3] or psi_type == 3:
        return dwtmtx(n, display = False)

    if psi_type == PSI_NAMES[4] or psi_type == 4:
        return hwtmtx(n, display = False)

    if psi_type == PSI_NAMES[5] or psi_type == 5:
        return rvsmtx(n, display = False)


def Generate_PSIs(n, savepath, display = True): # "PSIS.pkl"

    # psi_names = ['Identity Matrix', 'DCT', 'DFT', 'DWT', 
    # 'Hadamard-Walsh Matrix', 'Random Orthogonal Matrix']
    psi_mtxs = [
        np.identity(n), # the identitiy matrix, just for comparison
        dctmtx(n,n, display = display),
        dftmtx(n, display = display),
        dwtmtx(n, display = display)[0],
        hwtmtx(n, display = display)[0],
        rvsmtx(n, display = display)
    ]

    PSIs = {}

    for idx, psi in enumerate(PSI_NAMES):
        PSIs[psi] = psi_mtxs[idx]

    filehandler = open(savepath,"wb")
    pickle.dump(PSIs, filehandler)
    filehandler.close()

    print('saved to ' + savepath)

    for key in PSIs:
        
        # for some transformations, PSI dimension may differ. e.g. HWT requires 2**n and DWT requires even number
        n = PSIs[key].shape[0]
        _, OMEGA = GetSensingMatrix(n)
        
        print("Mutual Coherence (" + key, ", OMEGA) : ", Mutual_Coherence(PSIs[key], OMEGA))

def Mutual_Coherence(A,B):
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

def Sparsity(x, flavor = 'gini'):
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
    
def CompareSparsityFlavors():

    s1 = []
    s2 = []
    s3 = []

    for i in range(100):
        v =  [0]*i + [1]*(100-i) 
        s1.append(Sparsity (v, flavor = 'L0_inverse'))
        s2.append(Sparsity (v, flavor = 'gini_v1'))
        s3.append(Sparsity (v, flavor = 'gini_v2'))


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


def Analyze_Sparsity (x, PSIs):
    '''
    Parameters
    ----------
    x : a 1D signal. Will convert to shape (1,n)
    PSIs : a dict of PSI matrices
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
        
        PSI_H = PSI.conj().T
        PSI_INV = np.linalg.pinv(PSI)
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
        plt.plot(z, color='gray')
        title = key
        if key in PSI_NAMES:
            psi_idx = PSI_NAMES.index(key)
            title = PSI_LONGNAMES[ psi_idx ] + ' (' + key + '), MC = ' + PSI_MC[ psi_idx ]
        plt.title( title )
        # plt.axis('off')
        
        plt.subplot(rows,2,2*idx+2)
        plt.scatter(thresholds, rs, color='gray')
        plt.title('AUC = ' + str(round(auc,3)) + ', $r_{0.02max}$ = ' + str(round(r,3)) + ', gini = ' + str(round(Sparsity(z),3)))
        pylab.xlim([0,0.02])
        pylab.ylim([0,1.0])
        plt.xticks(np.arange(0.0, 0.021, 0.005))

    plt.show()    
    matplotlib.rcParams.update({'font.size': 12})

def GetSensingMatrix(n, k = 0.2, s = None):

    '''
    Parameters
    ----------
    n : signal dimensions
    k : sampling ratio
    s : random seed. Specify a seed value for TVSM senarios.
    '''
    if s is not None:
        np.random.seed(s)
    
    OMEGA = np.zeros((n,n))
    pm = np.random.permutation(np.arange(0, n))
    for i in range(n):
        OMEGA[i, pm[i]] = 1
    
    PHI = OMEGA[:int(n*k)]
    return PHI, OMEGA

def SensingWithPHI(x, PHI): # x is a list
    assert (x.shape[0] == PHI.shape[1])
    return PHI @ x

def Sensing(x, k = 0.2):      
    '''
    信号采样，发送端处理

    Parameters
    ----------
    k : (0,1]. sampling ratio.

    Return
    ------
    xs : sampled signal
    r : sampling indices
    '''
    
    n = len(x)
    pm = np.random.permutation(np.arange(1, n))
    r = pm[:int(n*k)]
    xs = x[r]

    return xs, r

from scipy.sparse import csr_matrix
def MeasurementMatrix (N, r, t = 'DCT'): 

    '''
    Construct the measurement matrix. 
    Currently only support DCT and DFT.
    '''
    
    kn = len(r)    
    a = np.zeros(N)    
        
    # Suppose the first number in the permutation is 42. Then choose the 42th
    # elements in a matrix full of zeros and change it to 1. Apply a discrete
    # cosine transformation to this matrix, and add the result to other matrix
    # Adelta as a new row.
    # A = np.zeros((kn, N))

    A = np.zeros ((kn, N)) #csr_matrix, dtype='float16') # reduce memory use. numpy uses float64 as default dtype.

    for i, j in enumerate(r):
        a[j] = 1
        if t == 'IDM' or t == 0:
            A[i] = a # .astype(np.float16)
        elif t == 'DCT' or t == 1:
            A[i] = cv2.dct(a).flatten() # .astype(np.float16)
        elif t == 'DFT' or t == 2:
            A[i] = cv2.dft(a).flatten() # .astype(np.float16)
        a[j] = 0
    
    ###### ALTERNATIVE IMPLEMENTATION ######
    # phi = np.zeros((k, N)) # Sampling matrix, kxN    
    # for i, j in enumerate(r):
    #    phi[i,j] = 1
    # psi = dctmtx(N,N) # memory costly
    # A = phi*psi
    
    return A # .toarray()

def Recovery (A, xs, t = 'DCT', PSI = None, solver = 'LASSO', fast_lasso = False, display = True):
    '''    
    Solve the optimization problem A@z = y
    
    Parameters
    ----------
    A : measurement matrix
    y : = xs, sampled signal
    alpha[obselete] : controls LASSO sparsity. Bigger alpha will return sparser result.
                      Now we use LassoCV. This param is no longer needed.
    t : IDM, DCT, DFT, etc.
    PSI : For adpative transforms, users need to pass in the PSI basis. For non-adaptive ones, leave as none.
    solver : 'LASSO' or 'OMP'
    '''

    if solver == 'LASSO':

        alphas = None # automatic set alphas
        if  t == PSI_NAMES[5] or t == PSI_LONGNAMES[5] or t == 'rom' or \
            t == 'DFT' or t == 'dft' or \
            t == 'DWT' or t == 'dwt' or \
            t == 'LDA' or t == 'lda':
            alphas = [0.0001, 0.00001] # set empirical alpha values. LDA needs smaller alpha

        # 实测并未发现两种模式的运行时间差异
        if fast_lasso:
            lasso = LassoCV(alphas = alphas, selection = 'random', tol = 0.001) # ‘random’ often leads to significantly faster convergence especially when tol is higher than 1e-4.
        else:
            lasso = LassoCV(alphas = alphas, selection = 'cyclic')

        lasso.fit(A, xs)
        z = lasso.coef_

    else: # 'OMP'

        # plot the noise-free reconstruction
        omp = OrthogonalMatchingPursuit() # n_nonzero_coefs default 10%.
        omp.fit(A, xs)
        z = omp.coef_
        # print('OMP score:', omp.score(A, xs)) # Return the coefficient of determination R^2 of the prediction.
    
    if t == PSI_NAMES[0] or t == PSI_LONGNAMES[0] or t == 'idm':
        z = np.linalg.pinv(A) @ xs
        xr = z
    elif  t == PSI_NAMES[1] or t == PSI_LONGNAMES[1] or t == 'dct':
        xr = cv2.idct(z)
    elif t == PSI_NAMES[2] or t == PSI_LONGNAMES[2] or t == 'dft':
        xr = cv2.idft(z)
    elif t == PSI_NAMES[3] or t == PSI_LONGNAMES[3] or t == 'dwt' or \
        t == PSI_NAMES[4] or t == PSI_LONGNAMES[4] or t == 'hwt' or \
        t == PSI_NAMES[5] or t == PSI_LONGNAMES[5] or t == 'rom' or \
        t == 'EBP' or t == 'ebp' or t == 'LDA' or t == 'lda': # these are adaptive bases
        xr = (PSI @ z).T
    else:
        print ('unsupported transform type: ', t)
        return

    if display:

        plt.figure(figsize = (10,3))
        # print ('non-zero coef: ', np.count_nonzero(z))
        # print ('sparsity: ', 1 - np.count_nonzero(z) / len(z) )
        biggest_lasso_fs = (np.argsort(np.abs(z))[-200:-1])[::-1] # take last N item indices and reverse (ord desc)
        plt.plot(np.abs(z))
        plt.title('abs (z)')
        plt.show()

        plt.figure(figsize = (10,3))
        plt.plot(xr)
        plt.title('Reconstructed Signal (xr)')
        plt.show()

    return z, xr

def Sensing_n_Recovery(x, k = 0.2, t = 'DCT', solver = 'LASSO', fast_lasso = False, display = True):
    
    '''
    Parameters
    ----------
    k : sampling ratio/percentage. 
        In normal cases (k<1), Az = xs has more unknowns than equations. For the extreme case of k = 1, CS sampling is degraded to a random shuffling. The linear system 'Az = xs' will have equal unknowns and equations, and there is only one solution. The reconstructed xr will be identical to the original x.
    '''

    VECTORIZATION = False    
    if VECTORIZATION:
        PHI, OMEGA = GetSensingMatrix(len(x), k) 
        xs = PHI @ x
        PSI = Generate_PSI(len(x), t)
        # print(xs)
        # print(PSI)
        A = PHI @ PSI
        
    else:

        # imgdata, w, h = get_img_data('symmetry.jpg')
        xs, r = Sensing(x, k)
        A = MeasurementMatrix (len(x), r, t = t)

    z,xr = Recovery (A, xs, t = t, solver = solver, fast_lasso=fast_lasso, display = False)

    if display: 

        matplotlib.rcParams.update({'font.size': 20})

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32,6))
        ax[0].plot(z, color = 'gray')
        ax[0].set_title('recovered latent representation (z)')

        ax[1].plot(xr, color = 'gray')
        ax[1].set_title('reconstructed signal (xr)')

        fig.tight_layout()
        plt.show()

        matplotlib.rcParams.update({'font.size': 12})

    return z, xr

def Dataset_Sensing_n_Recovery (X, y = None, k = 0.2, t = 'DCT', solver = 'LASSO', fast_lasso = False, display = 'all'):
    
    '''
    display : 'all' - display all samples
              'first' - display only the first sample
              'none' - don't display
    '''

    print('\n\n===== Ψ = '  + t + ', k =' + str(round(k,2)) + ' ======\n')

    Z = np.zeros(X.shape)
    Xr = np.zeros(X.shape)

    if display == 'all':
        b = True
    elif display == 'none':
        b = False
    
    if (k > 1.0): # when k > 1.0, return original signal directly
        Xr = X 
        Z = X
    else:
        for i in range(X.shape[0]):            
            x = X[i] # X[i,:].ravel().tolist()[0] # get the i-th sample

            if display == 'first':
                b = (i == 0)

            if b:
                print('Sample ' + str(i+1))

            xr, z = Sensing_n_Recovery(x, k, t, solver = solver, display = b)
            Z[i,:] = list(z)
            Xr[i,:]= xr #[:,0]

    pca = PCA(n_components=None) # n_components == min(n_samples, n_features) - 1. But we will use the first 2 components
    Z_pca = pca.fit_transform(Z)
    # plotComponents2D(Z_pca, y, labels, use_markers = False, ax=ax[0])
    # ax[0].title.set_text('2D-Visualization')

    pca = PCA(n_components=None)
    Xr_pca = pca.fit_transform(Xr)
    
    # For classification problem, continue to analyze with ANOVA and MANOVA
    if y is None:

        plt.scatter(Xr_pca[:,0], Xr_pca[:,1], s=40, 
        edgecolors = 'black', alpha = .4)
        plt.title('2D Visualization')
        plt.show()

    else:

        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24,4))

        pca = PCA(n_components=None)
        X_pca = pca.fit_transform(X)

        plotComponents2D(X_pca, y = y, use_markers = False, ax=ax[0]) 
        ax[0].title.set_text('PCA Visualization (X)')

        plotComponents2D(Xr_pca, y = y, use_markers = False, ax=ax[1]) 
        ax[1].title.set_text('PCA Visualization (Xr)')

        Xc1s = []
        Xc2s = []
        title = ''
        for c in set(y): 
            # print(Xr.shape, Z.shape, Xr_pca.shape, Z_pca.shape, y.shape)
            Xc = Xr_pca[y == c]
            yc = y[y == c]
            Xc1s.append(list(np.asarray(Xc[:,0]).reshape(1,-1)[0])) # First PC of Class c
            Xc2s.append(list(np.asarray(Xc[:,1]).reshape(1,-1)[0])) # Second PC of Class c

        #### ANOVA ####

        ax[2].title.set_text('PC1')
        ax[2].boxplot(Xc1s, notch=False) # plot 1st PC of all classes
        f,p= scipy.stats.f_oneway(Xc1s[0], Xc1s[1]) # equal to ttest_ind() in case of 2 groups
        print("f={},p={}".format(f,p))
        # dict_anova_p1[idx] = p
        
        ax[3].title.set_text('PC2')
        ax[3].boxplot(Xc2s, notch=False)
        f,p= scipy.stats.f_oneway(Xc2s[0], Xc2s[1])
        print("f={},p={}".format(f,p))
        
        #### MANOVA ####

        import statsmodels
        from statsmodels.base.model import Model
        from statsmodels.multivariate.manova import MANOVA
        import pandas as pd

        df = pd.DataFrame({'PC1':Xr_pca[:,0],'PC2':Xr_pca[:,1],'y':y})

        mv = MANOVA.from_formula('PC1 + PC2 ~ y', data=df) # Intercept is included by default.
        print(mv.endog_names)
        print(mv.exog_names)
        r = mv.mv_test()
        print(r)
        #dict_manova_p[idx] = str(r.results['y']['stat']['Pr > F'])
        
        #### Classifier ACC
        from sklearn.model_selection import cross_val_score
        from sklearn import svm
        
        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, Xr, y, cv=4)
        print('SVC ACC: ', scores.mean())      
        plt.show()
    
        # return scores.mean(), str(r.results['y']['stat']['Pr > F'])

        #### Mutlivariate KLD

        n_comp = 2
        print('\n ------ Class-wise Multivarate KLD on ' + str(n_comp) + ' PCA components -----')
        
        dists = []
        
        for XPCA in [ X_pca[:,:n_comp], Xr_pca[:,:n_comp] ]:

            dist = {}

            for c in set(y):    
                Xc = XPCA[y == c]
                yc = y[y == c]
                
                # np.cov()
                # ddof=1 will return the unbiased estimate, even if both fweights and aweights are specified, and ddof=0 will return the simple average. 
                # There is exactly 1 difference between np.cov(X) and np.cov(X, ddof=0) which is the bias step. With ddof=1 the dot is divided by 2 (X.shape[1] - 1), while with ddof=0 the dot is divided by 3 (X.shape[1] - 0).
                dist[c] = ( Xc.mean(axis = 0), np.cov(Xc, rowvar = False) )
            
            dists.append(dist)


        for c in set(y):

            mkld = Multivarate_KLD(dists[0][c], dists[1][c])
            print('Multivarate KLD between the orignal and reconstructed signals ( y = ' + str(c) +  '): ', round(mkld,3))
    
        print('\n ------ Class-wise Multivarate KLD on LDA component -----')
        
        X_lda = LinearDiscriminantAnalysis().fit_transform(X,y)[:,0].flatten()
        Xr_lda = LinearDiscriminantAnalysis().fit_transform(Xr,y)[:,0].flatten()

        for c in set(y):    
            Xc1 = X_lda[y == c]
            Xc2 = Xr_lda[y == c]

        ukld = Univariate_KLD( (Xc1.mean(), Xc1.std()),  (Xc2.mean(), Xc2.std()) )
        print('KLD between the orignal and reconstructed signals ( y = ' + str(c) +  '): ', round(ukld,3))
    
    return Z, Xr

def Univariate_KLD(p, q):

    # p is target distribution
    return np.log(q[1] / p[1]) + (p[1] ** 2 + (p[0] - q[0]) ** 2) / (2 * q[1] ** 2) - 0.5


def Multivarate_KLD(p, q):

    a = np.log(np.linalg.det(q[1])/np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)

def GridSearch_Sensing_n_Recovery(x, PSIs, ks = [0.1, 0.2, 0.5, 1.001], solver = 'LASSO'):

    plt.figure(figsize=(40, len(PSIs)*5))
    rows = len(PSIs) + 1
    matplotlib.rcParams.update({'font.size': 24})    
    COLS = 2 + len(ks)
    RMSES = []

    for idx, key in enumerate(PSIs):     

        PSI = PSIs[key]
        
        # padded version
        xe = np.copy(x)
        
        # pad x with zero if necessary
        if (len(x) < len(PSI)): # the HWT case
            xe = np.zeros(len(PSI))
            xe[:len(x)] = x     
        if len(xe) % 2 == 1:
            xe = xe[:-1] # xe = np.append(xe, 0) # make even length, as required by some transform, e.g., DCT
            PSI = PSI[:-1,:-1] # align with xe
        
        psi_name = key
        
        PSI_H = PSI.conj().T
        PSI_INV = np.linalg.pinv(PSI)    
        assert np.allclose(PSI_H, PSI_INV) # Unitary Matrix
        
        #print(PSI_H)
        #print(PSI_INV)
        # theoretically, PSI_H == PSI_INV
        z = PSI @ xe
        # print((xx @ PSI_H).shape)
        # print("z =", z[:10])
        MAX = np.max(np.abs(z)) # abs(max(z, key=abs))
        thresholds = np.array(range(100)) / 50000
        
        rs =[]
        for threshold in thresholds:
            rs.append((np.abs(np.array(z)) <= threshold * MAX).sum() / len(z))
            
        auc = 0
        for i in range(1000):
            auc += (np.abs(np.array(z)) <= (i+1)/1000 * MAX).sum() / len(z)        
        auc = auc/1000
        
        r = (np.abs(np.array(z)) <= 0.001 * MAX).sum() / len(z)  #   # use 0.001 MAX ABS as threshold # sys.float_info.epsilon
        
        
        ########## plot ###########
    
        plt.subplot(rows,COLS,COLS*idx+1)    
        #if key == 'EBP':      
        #    ebp_dim = np.linalg.matrix_rank(X[1:,:]) 
        #    plt.title('\n'+psi_name + ' basis')
        #    plt.imshow(PSI[:ebp_dim, :ebp_dim], interpolation='nearest', cmap=cm.Greys_r) # Q2
        if PSI.dtype == 'complex': # DFT
            plt.title('\n'+psi_name + ' basis(phase)')
            plt.imshow(np.angle(PSI), interpolation='nearest', cmap=cm.Greys_r)
        else:
            plt.title('\n'+psi_name + ' basis')
            plt.imshow(PSI, interpolation='nearest', cmap=cm.Greys_r)
        plt.axis('off')
            
        plt.subplot(rows,COLS,COLS*idx+2)
        plt.plot(z, color='gray')
        if idx == 0:
            plt.title('z (latent space)') # '\n'+psi_name + 
        if psi_name == 'EBP':
            pylab.ylim([-abs(max(z,key=abs)), abs(max(z,key=abs))])
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')
        
        rmses = []
        
        for kidx, k in enumerate(ks):
        
            ########## sensing #############

            # either works fine
            if True:
                PHI, OMEGA = GetSensingMatrix(len(xe), k)            
                xs = PHI @ xe
                pidx = np.argmax(PHI, axis = 1)
            else:
                xs, pidx = Sensing(xe, k)
                PHI = np.zeros((len(pidx), len(xe)))
                for i,j in enumerate(pidx):
                    PHI[i, j] = 1

            ####### reconstruction ##########
            
            A = MeasurementMatrix(len(xe), pidx, psi_name)
            W = None
            if (psi_name == 'HWT' or psi_name == 'DWT' or psi_name == 'ROM' or \
                psi_name == 'EBP' or psi_name == 'LDA'):
                W = PSI
                A = PHI @ W  

            else:
                A = MeasurementMatrix(len(xe), pidx, psi_name)
                W = None

            z, xr = Recovery (A, xs, psi_name, display = False, PSI = W, solver = solver) # lower k needs bigger L1. k 0.1 - L1 0.1, k 0.01, L1 - 10
            
            _, _, rmse = calculate_recon_error(xe.reshape(1, -1), xr.reshape(1, -1)) #(np.matrix(xe), np.matrix(xr))        
            rmses.append(rmse)
            
            plt.subplot(rows,COLS,COLS*idx+2+1+kidx)
            plt.plot(xr[:len(x)], label = 'RMSE ' + str( round(rmse, 3) ), color='gray') # cut the first n points, as in HWT, xr is padded.
            if idx == 0:
                plt.title('$x_r$ ($k$=' + str(round(k, 2)) + ', $n_s$=' + str(len(xs)) + ')')
            plt.xticks([])
            plt.yticks([])
            # plt.legend()
            
        RMSES.append(rmses)
        
        #plt.subplot(rows,3,3*idx+3)
        #plt.scatter(thresholds, rs, color='gray')
        #plt.title('\nAUC = ' + str(round(auc,3)) + ', s = ' + str(round(r,3))) # $s_{2\%}$ 
        #pylab.xlim([0,0.002])
        #pylab.ylim([0,1.0])
        #plt.xticks(np.arange(0.0, 0.0021, 0.0005))
        
        #plt.subplots_adjust(top=1., left=0., right=1., bottom=0.)

    matplotlib.rcParams.update({'font.size': 12})

    print('\n-------------\nThe following are dynamic properties of each PSI:')
    DynamicProperty(RMSES, PSIs, np.round(ks,2))
    return RMSES

def DynamicProperty(RMSES, PSIs, ks, repeat = 1):

    assert (len(RMSES) == len(PSIs))
    matplotlib.rcParams.update({'font.size': 16})
    
    for i, key in enumerate(PSIs):   
            
        plt.figure(figsize=(8,6))
        
        plt.scatter(ks, RMSES[i], c='gray', s = 70, label = key)
        plt.plot(ks, RMSES[i], c='gray')
        plt.xlabel('k')
        plt.ylabel('RMSE')
        
        plt.legend()
        plt.show()

    matplotlib.rcParams.update({'font.size': 12})


def Simulate_ECG(bpm = 60, time_length = 10, display = True):

    '''
    Parameters
    ----------
    bpm : Simulated Beats per minute rate. For a health, athletic, person, 60 is resting, 180 is intensive exercising
    time_length : Simumated length of time in seconds
    '''

    bps = bpm / 60
    capture_length = time_length

    # Caculate the number of beats in capture time period 
    # Round the number to simplify things
    num_heart_beats = int(capture_length * bps)


    # The "Daubechies" wavelet is a rough approximation to a real,
    # single, heart beat ("pqrst") signal
    pqrst = signal.wavelets.daub(10)

    # Add the gap after the pqrst when the heart is resting. 
    samples_rest = 10
    zero_array = np.zeros(samples_rest, dtype=float)
    pqrst_full = np.concatenate([pqrst,zero_array])


    # Concatonate together the number of heart beats needed
    ecg_template = np.tile(pqrst_full , num_heart_beats)

    # Add random (gaussian distributed) noise 
    noise = np.random.normal(0, 0.01, len(ecg_template))
    ecg_template_noisy = noise + ecg_template


    if display:

        # Plot the noisy heart ECG template
        plt.figure(figsize=(int(time_length * 2), 3))
        plt.plot(ecg_template_noisy)
        plt.xlabel('Sample number')
        plt.ylabel('Amplitude (normalised)')
        plt.title('Heart ECG Template with Gaussian noise')
        plt.show()
        
    return ecg_template_noisy

def dct_lossy_signal_compression(x, percent = 99):    

    dst = cv2.dct(x)
    lossy_dct = dst.copy()

    threshold = np.percentile(dst, percent) # compute the percentile(s) along a flattened version of the array.
    for i, element in enumerate(abs(dst)):
        if element < threshold:
            lossy_dct[i] = 0

    print ('non-zero elements: ', np.count_nonzero(lossy_dct))
    lossy_idct = cv2.idct(lossy_dct)
    print('Compression ratio = ', 100-percent, '%')

    plt.figure(figsize=(9,9))

    plt.subplot(311)
    plt.plot(x, 'gray')
    plt.title('original signal')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(312)
    plt.plot(abs(lossy_dct),'gray')
    plt.title('lossy/sparse DCT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(313)
    plt.plot(lossy_idct, 'gray')
    plt.title('recovered signal (IDCT)')
    plt.xticks([]), plt.yticks([])

    return lossy_idct

def dft_lossy_signal_compression(x, percent = 99):    

    dst = cv2.dft(x)
    lossy_dft = dst.copy()

    threshold = np.percentile(dst, percent) # compute the percentile(s) along a flattened version of the array.
    for i, element in enumerate(abs(dst)):
        if element < threshold:
            lossy_dft[i] = 0

    print ('non-zero elements: ', np.count_nonzero(lossy_dft))
    lossy_idft = cv2.idft(lossy_dft)
    print('Compression ratio = ', 100-percent, '%')

    plt.figure(figsize=(9,9))

    plt.subplot(311)
    plt.plot(x, 'gray')
    plt.title('original signal')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(312)
    plt.plot(abs(lossy_dft),'gray')
    plt.title('lossy/sparse DCT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(313)
    plt.plot(lossy_idft, 'gray')
    plt.title('recovered signal (IDCT)')
    plt.xticks([]), plt.yticks([])

    return lossy_idft

def Frequency_Analysis(x):

    fourier = np.fft.fft(x)
    n = len(x)
    FFT = abs(fourier)
    freq = np.fft.fftfreq(n, d=1) # d - Sample spacing (inverse of the sampling rate). Defaults to 1.
    plt.plot(freq, FFT)


############ Below are TVSM functions ##############

import time

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

############ Below are image-related functions ##############


def get_img_data(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # input a square image
    w,h = img.shape
    return img.reshape(w*h), w, h # expand to 1D array

def img_dct(path):

    # Second argument is a flag which specifies the way image should be read.
    #  cv2.IMREAD_COLOR (1) : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    #  cv2.IMREAD_GRAYSCALE (0) : Loads image in grayscale mode
    #  cv2.IMREAD_UNCHANGED (-1): Loads image as such including alpha channel
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    # imgdata = img.reshape(1, w*h) # expand to 1D array

    mtx = dctmtx(w, h, True)

    # dctmtx * dctmtx' = E

    E = np.dot(mtx, np.transpose(mtx))
    plt.imshow(E, cmap=cm.Greys_r)
    plt.title('DCT @ DCT.T = I')
    plt.axis('off')
    plt.show()

    dst = np.dot(mtx , img)
    dst = np.dot(dst, np.transpose(mtx))
    dst2 = cv2.dct(img) # Method 2 - using opencv 

    assert np.allclose(dst, dst2)

    # IDCT to reconstruct image

    # Method 1
    img_r = np.dot(np.transpose(mtx) , dst)
    img_r = np.dot(img_r, mtx)

    # Method 2 - opencv
    img_r2 = cv2.idct(dst2)

    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(232)
    plt.imshow(np.log(abs(dst)),'gray')
    plt.title('DCT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(233)
    plt.imshow(img_r, 'gray')
    plt.title('IDCT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(234)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(np.log(abs(dst2)),'gray')
    plt.title('DCT(cv2)')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(img_r2,'gray')
    plt.title('IDCT(cv2)')
    plt.axis('off')

    plt.show()

    v = cv2.Laplacian(np.log(abs(dst)), cv2.CV_64F).var() # Blur Detection using the variance of the Laplacian method
    print('Laplacian var (Blur Detection) of DCT = ', round(v,3) )
    v2 = cv2.Laplacian(np.log(abs(dst2)), cv2.CV_64F).var() # Blur Detection using the variance of the Laplacian method
    print('Laplacian var (Blur Detection) of DCT (cv2) = ', round(v2,3) )


def img_dft(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    s = min(w,h)
    img = img[:s,:s]
    print('image cropped to ', str(s), " x ", str(s) )
    # imgdata = img.reshape(1, s*s) # expand to 1D array

    mtx = dftmtx(s, True)

    E = np.dot(mtx, np.transpose(mtx))
    plt.imshow(np.abs(E), cmap=cm.Greys_r)
    plt.title('DFT @ DFT.T = I')
    plt.axis('off')
    plt.show()

    dst = np.dot(mtx , img)
    dst = np.dot(dst, np.transpose(mtx))
    dst2 = cv2.dft(img) # Method 2 - using opencv 

    # assert np.allclose(dst, dst2)

    # IDCT to reconstruct image

    # Method 1
    img_r = np.dot(np.transpose(mtx) , dst)
    img_r = np.abs( np.dot(img_r, mtx) )

    # Method 2 - opencv
    img_r2 = cv2.idft(dst2)

    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(232)
    plt.imshow(np.log(abs(dst)),'gray')
    plt.title('DFT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(233)
    plt.imshow(img_r, 'gray')
    plt.title('IDFT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(234)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(np.log(abs(dst2)),'gray')
    plt.title('DFT(cv2)')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(img_r2,'gray')
    plt.title('IFCT(cv2)')
    plt.axis('off')

    plt.show()

    v = cv2.Laplacian(np.log(abs(dst)), cv2.CV_64F).var() # Blur Detection using the variance of the Laplacian method
    print('Laplacian var (Blur Detection) of DFT = ', round(v,3) )
    v2 = cv2.Laplacian(np.log(abs(dst2)), cv2.CV_64F).var() # Blur Detection using the variance of the Laplacian method
    print('Laplacian var (Blur Detection) of DFT (cv2) = ', round(v2,3) )

def img_dwt(path, wavelet = 'db3', flavor = 1):
    '''
    flavor = 1: use pywt dwt2 and idwt2
    flavor = 2: use wavelet ortho basis matrix
    '''
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    # imgdata = img.reshape(1, w*h) # expand to 1D array

    if (flavor == 1):

        img_dwt = pywt.dwt2(img, wavelet)
        img_idwt = pywt.idwt2(img_dwt, wavelet)

        plt.figure(figsize=(9,6))

        plt.subplot(231)
        plt.imshow(img, 'gray')
        plt.title('original image')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(232)
        plt.imshow(np.log(abs(img_dwt[0])), 'gray')
        plt.title('DWT LL')
        plt.xticks([]), plt.yticks([])

        plt.subplot(233)
        plt.imshow(np.log(abs(img_dwt[1][0])), 'gray')
        plt.title('DWT LH')
        plt.xticks([]), plt.yticks([])

        plt.subplot(234)
        plt.imshow(np.log(abs(img_dwt[1][1])), 'gray')
        plt.title('DWT HL')
        plt.xticks([]), plt.yticks([])

        plt.subplot(235)
        plt.imshow(np.log(abs(img_dwt[1][2])), 'gray')
        plt.title('DWT HH')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(236)
        plt.imshow(img_idwt, 'gray')
        plt.title('IDWT')
        plt.xticks([]), plt.yticks([])

        plt.show()

    else:

        mtx, _ = dwtmtx(w, wavelet, False)
        mtx_h = mtx.conj().T #np.linalg.pinv(mtx)
        img_dwt = mtx_h @ img #pywt.dwt2(img, 'db3')
        img_idwt = mtx @ img_dwt #pywt.idwt2(img_dwt, 'db3')

        plt.figure(figsize=(9,3))

        plt.subplot(131)
        plt.imshow(img, 'gray')
        plt.title('original image')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(132)
        plt.imshow(np.log(abs(img_dwt)), 'gray')
        plt.title('DWT')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(133)
        plt.imshow(img_idwt, 'gray')
        plt.title('IDWT')
        plt.xticks([]), plt.yticks([])

        plt.show()

        # v = cv2.Laplacian(np.log(abs(img_dwt)), cv2.CV_64F).var()
        # print('Laplacian var of DWT = ', v)  #nan

def print_wavelet_families():

    for family in pywt.families():
        print("%s family: " % family + ', '.join(pywt.wavelist(family)))


def dct_lossy_image_compression(path, percent = 99):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    # imgdata = img.reshape(1, w*h) # expand to 1D array

    dst = cv2.dct(img)
    lossy_dct = dst.copy()

    threshold = np.percentile(dst, percent) # compute the percentile(s) along a flattened version of the array.
    for (x, y), element in np.ndenumerate(abs(dst)):
        if element < threshold:
            lossy_dct[x,y] = 0

    print ('non-zero elements: ', np.count_nonzero(lossy_dct))
    lossy_idct = cv2.idct(lossy_dct)
    print('Compression ratio = ', 100-percent, '%')

    plt.figure(figsize=(9,3))

    plt.subplot(131)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(np.log(abs(lossy_dct)),'gray')
    plt.title('lossy/sparse DCT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(lossy_idct, 'gray')
    plt.title('IDCT')
    plt.xticks([]), plt.yticks([])

    return lossy_idct

def dft_lossy_image_compression(path, percent = 99):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    # imgdata = img.reshape(1, w*h) # expand to 1D array

    dst = cv2.dft(img)
    lossy_dft = dst.copy()

    threshold = np.percentile(dst, percent)
    for (x, y), element in np.ndenumerate(abs(dst)):
        if element < threshold:
            lossy_dft[x,y] = 0

    print ('non-zero elements: ', np.count_nonzero(lossy_dft))
    lossy_idft = cv2.idft(lossy_dft)
    print('Compression ratio = ', 100-percent, '%')

    plt.figure(figsize=(9,3))

    plt.subplot(131)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(np.log(abs(lossy_dft)),'gray')
    plt.title('lossy/sparse DFT')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(lossy_idft, 'gray')
    plt.title('IDFT')
    plt.xticks([]), plt.yticks([])

    return lossy_idft


def Image_Sensing_n_Recovery(path, k = 0.2, alpha = 0.01, t = 'DCT', fast_lasso = False):

    x, w, h = get_img_data(path)
    z, xr = Sensing_n_Recovery(x, k = k, alpha = alpha, t = t, fast_lasso=fast_lasso)

    img_r = xr.reshape((w,h))

    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    ax.imshow(x.reshape((w,h)), 'gray')
    ax.axis('off')
    ax.set_title('original image')

    ax = fig.add_subplot(132)
    if t == 'IDM':
        dst = x.reshape((w,h))
    elif t == 'DCT':
        dst = cv2.dct(x.reshape((w,h)))
    else:
        dst = cv2.dft(x.reshape((w,h)))
    ax.imshow(np.log(abs(dst)), interpolation='nearest', cmap=cm.Greys_r)
    ax.axis('off')
    ax.set_title(t)

    ax = fig.add_subplot(133)
    ax.imshow(img_r, 'gray')
    ax.axis('off')
    ax.set_title('recovered image')

    plt.show()

    return x, z, xr, w, h