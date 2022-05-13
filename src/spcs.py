import matplotlib.pyplot as plt
import pylab, matplotlib
import numpy as np
from tqdm import tqdm
import time
import matplotlib.ticker as mticker
import cv2
import matplotlib.cm as cm
from sklearn.linear_model import Lasso
import pywt
from scipy.stats import ortho_group
import scipy
from scipy.linalg import hadamard
import math
from scipy.stats import ortho_group
import pickle
from sklearn.decomposition import PCA
import scipy.signal as signal
from plotComponents2D import *

import warnings
warnings.filterwarnings("ignore")

#region Performance Test. Compare NNRW with mainstream classifier models.

PSI_NAMES = ['Identity Matrix', 'DCT', 'DFT', 'DWT', 
'Hadamard-Walsh Matrix', 'Random Orthogonal Matrix']

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

def Analyze_PSIs(x, PSIs):
    '''
    Parameters
    ----------
    x : a 1D signal. Will convert to shape (1,n)
    PSIs : a dict of PSI matrices
    '''

    x = x.reshape((-1,1))
    print(x.shape)

    plt.figure(figsize=(24,48))
    rows = len(PSIs)
    matplotlib.rcParams.update({'font.size': 18})

    for idx, key in enumerate(PSIs):
        
        PSI = PSIs[key]        
        xx = np.copy(x)
        
        # pad x with zero if necessary
        if (x.shape[1] < PSI.shape[0]):
            xx = np.zeros((x.shape[0], PSI.shape[0]))
            xx[:x.shape[0],:x.shape[1]] = x        
        
        PSI_H = PSI.conj().T
        PSI_INV = np.linalg.pinv(PSI)
        #print(PSI_H)
        #print(PSI_INV)
        # theoretically, PSI_H == PSI_INV
        z = (xx @ PSI_H).ravel().tolist()
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
        plt.title(key)
        # plt.axis('off')
        
        plt.subplot(rows,2,2*idx+2)
        plt.scatter(thresholds, rs, color='gray')
        plt.title('AUC = ' + str(round(auc,3)) + ', $r_{0.02max}$ = ' + str(round(r,3)))
        pylab.xlim([0,0.02])
        pylab.ylim([0,1.0])
        plt.xticks(np.arange(0.0, 0.021, 0.005))    

    plt.show()    
    matplotlib.rcParams.update({'font.size': 12})


def GetSensingMatrix(n, k = 0.2):

    '''
    Parameters
    ----------
    n : signal dimensions
    k : sampling ratio
    '''
    
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
    '''
    
    kn = len(r)    
    a = np.zeros((1, N))    
        
    # Suppose the first number in the permutation is 42. Then choose the 42th
    # elements in a matrix full of zeros and change it to 1. Apply a discrete
    # cosine transformation to this matrix, and add the result to other matrix
    # Adelta as a new row.
    # A = np.zeros((kn, N))

    A = csr_matrix((kn, N)) #, dtype='float16') # reduce memory use. numpy uses float64 as default dtype.

    for i, j in enumerate(r):
        a[0,j] = 1
        if t == 'DFT':
            A[i, :] = cv2.dft(a) # .astype(np.float16)
        elif t == 'DFT':
            A[i, :] = cv2.dct(a) # .astype(np.float16)
        a[0,j] = 0
    
    ###### ALTERNATIVE IMPLEMENTATION ######
    # phi = np.zeros((k, N)) # Sampling matrix, kxN    
    # for i, j in enumerate(r):
    #    phi[i,j] = 1
    # psi = dctmtx(N,N) # memory costly
    # A = phi*psi
    
    return A

def Recovery (A, y, alpha = 0.01, t = 'DCT', fast_lasso = False, display = True):
    '''
    Solve the optimization problem A@z = y
    
    Parameters
    ----------
    A : measurement matrix
    y : = xs, sampled signal
    alpha : controls LASSO sparsity. Bigger alpha will return sparser result.
    t : dct or dft
    '''

    # 实测并未发现两种模式的运行时间差异
    if fast_lasso:
        lasso = Lasso(alpha = alpha, selection = 'random', tol = 0.001) # ‘random’ often leads to significantly faster convergence especially when tol is higher than 1e-4.
    else:
        lasso = Lasso(alpha = alpha, selection = 'cyclic')

    lasso.fit(A,y)

    if display:
        print ('non-zero coef: ', np.count_nonzero(lasso.coef_))
        print ('sparsity: ', 1 - np.count_nonzero(lasso.coef_) / len(lasso.coef_) )
        biggest_lasso_fs = (np.argsort(np.abs(lasso.coef_))[-200:-1])[::-1] # take last N item indices and reverse (ord desc)
        plt.plot(np.abs(lasso.coef_))
        plt.title('abs (z)')
        plt.show()

    z = lasso.coef_

    if t == 'DCT':
        xr = cv2.idct(z)
    else: # dft
        xr = cv2.idft(z)

    return z, xr

def Sensing_n_Recovery(x, k = 0.2, alpha = 0.01, t = 1, fast_lasso = False):
    
    if (t > 2):
        print('Current Version only supports DCT and DFT. Exiting...')
        return

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

    z,xr = Recovery (A, xs, alpha = alpha, t = t, fast_lasso=fast_lasso, display = False)

    return z, xr

def Dataset_Sensing_n_Recovery (X, y = None, k = 0.2, alpha = 0.01, t = 1, fast_lasso = False):
    
    print('===== k =', k, ', L1 =', alpha,' ======')

    Z = np.zeros(X.shape)
    Xr = np.zeros(X.shape)
    
    if (k > 1.0): # when k > 1.0, return original signal directly
        Xr = X 
        Z = X
    else:
        for i in range(X.shape[0]):
            x = X[i] # X[i,:].ravel().tolist()[0] # get the i-th sample
            xr, z = Sensing_n_Recovery(x, k, alpha, t)
            Z[i,:] = list(z)
            Xr[i,:]= xr #[:,0]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,4))

    pca = PCA(n_components=2) # keep the first 2 components
    Z_pca = pca.fit_transform(Z)
    # plotComponents2D(Z_pca, y, labels, use_markers = False, ax=ax[0])
    # ax[0].title.set_text('2D-Visualization')

    pca = PCA(n_components=2) # keep the first 2 components
    Xr_pca = pca.fit_transform(Xr)
    plotComponents2D(Xr_pca, y = y, use_markers = False, ax=ax[0])    
    ax[0].title.set_text('2D Visualization')
    
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

    ax[1].title.set_text('PC1')
    ax[1].boxplot(Xc1s, notch=False) # plot 1st PC of all classes
    f,p= scipy.stats.f_oneway(Xc1s[0], Xc1s[1]) # equal to ttest_ind() in case of 2 groups
    print("f={},p={}".format(f,p))
    # dict_anova_p1[idx] = p
    
    ax[2].title.set_text('PC2')
    ax[2].boxplot(Xc2s, notch=False)
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
    
    return scores.mean(), str(r.results['y']['stat']['Pr > F'])

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

############ Below are image-related functions ##############


def get_img_data(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
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

        mtx = dwtmtx(w, wavelet, False)
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
    if t == 'DCT':
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