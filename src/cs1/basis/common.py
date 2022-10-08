import math
import pywt
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy.stats import ortho_group
from .. import GetSensingMatrix,PSI_NAMES
from ..metrics import mutual_coherence


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
    
    mtx = scipy.linalg.hadamard(NN) / (2**(math.log(NN,2)/2))
    
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
        print('det(DWT_MTX) = ', round(np.linalg.det(mtx), 5))   

    return mtx, N

def print_wavelet_families():

    for family in pywt.families():
        print("%s family: " % family + ', '.join(pywt.wavelist(family)))

def DwtMcCurve():
    '''
    Plot the Mutual Coherence (MC) curve against dimensionality (N) for dwt
    '''

    Ns = [10, 20, 50, 100, 150, 200, 500, 1000, 2000] # dimensions
    mcs = []
    for N in Ns:
        
        _, OMEGA = GetSensingMatrix(N)
        psi, _ = dwtmtx(N, display = False)
        mc = mutual_coherence(psi, OMEGA)
        # print("N :", N, ".  : ", mc)
        mcs.append(mc)

    plt.plot(Ns, mcs)
    plt.title(r'Mutual Coherence (DWT, OMEGA)')
    plt.show()

    plt.plot(Ns, np.sqrt(Ns) * .8)
    plt.title(r'0.8 $\sqrt{n}$')
    plt.show()

def rvsmtx(N, display = True):
    '''
    Generate a uniformly distributed random orthogonal matrix by ortho_group.rvs (random variables)
    '''

    mtx = ortho_group.rvs(N) # uniformly distributed random orthogonal matrix
    
    if display:
        plt.figure()
        plt.imshow(mtx, interpolation='nearest', cmap=cm.Greys_r)
        plt.title('Random Orthogonal Matrix')
        plt.axis('off')
        plt.show()
    
    print('det(RVS_MTX) = ',  round(np.linalg.det(mtx), 5))    
    return mtx

def Generate_PSI(n, psi_type = 1):
    '''
    Generate specific transform basis.

    Parameter
    ---------
    psi_type : one of PSI_NAMES or its index, e.g., 'DCT', 1
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
    '''
    Generate PSIs (CS transform bases) and persist to a local pickle file.
    '''

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

    ''' # to avoid circular import. don't call metrics.

    for key in PSIs:
        
        # for some transformations, PSI dimension may differ. e.g. HWT requires 2**n and DWT requires even number
        n = PSIs[key].shape[0]
        _, OMEGA = cs.GetSensingMatrix(n)        
        print("Mutual Coherence (" + key, ", OMEGA) : ", metrics.mutual_coherence(PSIs[key], OMEGA))

    '''