import matplotlib.pyplot as plt
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

import warnings
warnings.filterwarnings("ignore")

#region Performance Test. Compare NNRW with mainstream classifier models.

def dctmtx(m, n, display = False):    
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

def Sensing(x, k = 0.2):      
    '''
    信号采样，发送端处理

    Parameters
    ----------
    a : (0,1]. sampling ratio.

    Return
    ------
    y : sampled signal
    r : sampling indices
    '''
    
    N = len(x)
    pm = np.random.permutation(np.arange(1, N))
    r = pm[:int(N*k)]
    y = x[r]

    return y, r

from scipy.sparse import csr_matrix
def MeasurementMatrix (N, r, t = 'dct'): 
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
        if t == 'dft':
            A[i, :] = cv2.dft(a) # .astype(np.float16)
        else:
            A[i, :] = cv2.dct(a) # .astype(np.float16)
        a[0,j] = 0
    
    ###### ALTERNATIVE IMPLEMENTATION ######
    # phi = np.zeros((k, N)) # Sampling matrix, kxN    
    # for i, j in enumerate(r):
    #    phi[i,j] = 1
    # psi = dctmtx(N,N) # memory costly
    # A = phi*psi
    
    return A

def Recovery (A, y, alpha = 0.01, t = 'dct'):

    # Solve the optimization problem A@z = y
    lasso = Lasso(alpha = alpha)
    lasso.fit(A,y)

    print ('non-zero coef: ', np.count_nonzero(lasso.coef_))
    print ('sparsity: ', 1 - np.count_nonzero(lasso.coef_) / len(lasso.coef_) )
    biggest_lasso_fs = (np.argsort(np.abs(lasso.coef_))[-200:-1])[::-1] # take last N item indices and reverse (ord desc)
    plt.plot(np.abs(lasso.coef_))
    plt.show()

    z = lasso.coef_

    if t == 'dct':
        xr = cv2.idct(z)
    else: # dft
        xr = cv2.idft(z)

    return z, xr

def Sensing_n_Recovery(x, k = 0.2, alpha = 0.01, t = 'dct'):

    # imgdata, w, h = get_img_data('symmetry.jpg')
    y, r = Sensing(x, k)
    A = MeasurementMatrix (len(x), r, t = t)
    z,xr = Recovery (A, y, alpha = alpha, t = t)

    return z, xr

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

def dwtmtx(N, wavelet = 'db3', display = True):
    mtx = np.zeros((N,N))  
    I = np.identity(N)
        
    for i in range(N):
        cA, cD = pywt.dwt(I[i,:], wavelet, pywt.Modes.periodization)
        mtx[i,:] = list(cA) + list(cD)
    
    if display:

        plt.figure()
        plt.imshow(mtx, interpolation='nearest', cmap=cm.Greys_r)
        plt.title(wavelet + ' wavelet sensing matrix')
        plt.axis('off')
        plt.show()
    
    print('det(DWT_MTX) = ', np.fabs(scipy.linalg.det(mtx)))    
    return mtx

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


def Image_Sensing_n_Recovery(path, k = 0.2, alpha = 0.01, t = 'dct'):

    x, w, h = get_img_data(path)
    z, xr = Sensing_n_Recovery(x, k = k, alpha = alpha, t = t)

    img_r = xr.reshape((w,h))

    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(131)
    ax.imshow(x.reshape((w,h)), 'gray')
    ax.axis('off')
    ax.set_title('original image')

    ax = fig.add_subplot(132)
    if t == 'dct':
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