import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import matplotlib.ticker as mticker
import cv2
import matplotlib.cm as cm
from sklearn.linear_model import Lasso

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


def img_dct(path):

    # Second argument is a flag which specifies the way image should be read.
    #  cv2.IMREAD_COLOR (1) : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    #  cv2.IMREAD_GRAYSCALE (0) : Loads image in grayscale mode
    #  cv2.IMREAD_UNCHANGED (-1): Loads image as such including alpha channel
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    imgdata = img.reshape(1, w*h) # expand to 1D array

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

def img_dft(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    s = min(w,h)
    img = img[:s,:s]
    print('image cropped to ', str(s), " x ", str(s) )
    imgdata = img.reshape(1, s*s) # expand to 1D array

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
