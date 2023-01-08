import matplotlib.pyplot as plt
import cv2
import pywt
import numpy as np
from matplotlib import cm
from .. import cs, basis

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

    mtx = basis.common.dctmtx(w, h, True)

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

    mtx = basis.common.dftmtx(s, True)

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
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(232)
    plt.imshow(np.log(abs(dst)),'gray')
    plt.title('DFT')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(233)
    plt.imshow(img_r, 'gray')
    plt.title('IDFT')
    plt.xticks([])
    plt.yticks([])
    
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
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(232)
        plt.imshow(np.log(abs(img_dwt[0])), 'gray')
        plt.title('DWT LL')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(233)
        plt.imshow(np.log(abs(img_dwt[1][0])), 'gray')
        plt.title('DWT LH')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(234)
        plt.imshow(np.log(abs(img_dwt[1][1])), 'gray')
        plt.title('DWT HL')
        plt.xticks([]), plt.yticks([])

        plt.subplot(235)
        plt.imshow(np.log(abs(img_dwt[1][2])), 'gray')
        plt.title('DWT HH')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(236)
        plt.imshow(img_idwt, 'gray')
        plt.title('IDWT')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    else:

        mtx, _ = basis.common.dwtmtx(w, wavelet, False)
        mtx_h = mtx.conj().T #np.linalg.pinv(mtx)
        img_dwt = mtx_h @ img #pywt.dwt2(img, 'db3')
        img_idwt = mtx @ img_dwt #pywt.idwt2(img_dwt, 'db3')

        plt.figure(figsize=(9,3))

        plt.subplot(131)
        plt.imshow(img, 'gray')
        plt.title('original image')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(132)
        plt.imshow(np.log(abs(img_dwt)), 'gray')
        plt.title('DWT')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(133)
        plt.imshow(img_idwt, 'gray')
        plt.title('IDWT')
        plt.xticks([])
        plt.yticks([])

        plt.show()

        # v = cv2.Laplacian(np.log(abs(img_dwt)), cv2.CV_64F).var()
        # print('Laplacian var of DWT = ', v)  #nan


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
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(np.log(abs(lossy_dct)),'gray')
    plt.title('lossy/sparse DCT')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(lossy_idct, 'gray')
    plt.title('IDCT')
    plt.xticks([])
    plt.yticks([])

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
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(np.log(abs(lossy_dft)),'gray')
    plt.title('lossy/sparse DFT')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(lossy_idft, 'gray')
    plt.title('IDFT')
    plt.xticks([])
    plt.yticks([])

    return lossy_idft

def dwt_lossy_image_compression(path, percent = 99, wavelet = 'db3', level = 3):
    '''
    Parameters
    ----------
    wavelet : a discrete wavelet type, e.g., haar, db3, etc. default = 'db3'
    '''

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float') # input a square image
    w,h = img.shape
    # imgdata = img.reshape(1, w*h) # expand to 1D array


    # levels = int(np.floor(np.log2(img.shape[0])))
    dec = pywt.wavedec2(img, wavelet, level=level)
    # print(aa.shape)
    # print(len(d2), 'x',  d2[0].shape) # ah,av,ad are tupled into d2
    # print(len(d1), 'x',  d1[0].shape) # h,v,d are tupled into d1

    # threshold = 40  # for haar, threshold 20 get a compress ratio = 4524 / (4524 + 11860) = 27.6%
    dst, coeff_slices = pywt.coeffs_to_array(dec)
    # print(dst.shape, dst.max(), dst.min())
    # print(coeff_slices)
    threshold = np.percentile(np.abs(dst), percent)
    # print( (np.abs(arr) >= threshold).sum(), (np.abs(arr) < threshold).sum() )
    lossy_dwt = pywt.threshold(dst, threshold, mode='soft', substitute=0)
    # plt.imshow(lossy_dst)
    # plt.show()
    nwc = pywt.array_to_coeffs(lossy_dwt, coeff_slices, output_format='wavedec2')
    lossy_idwt = pywt.waverec2(nwc, wavelet)
    # plt.imshow(imgr)
    # plt.show()

    print ('non-zero elements: ', np.count_nonzero(lossy_dwt))
    print('Compression ratio = ', 100-percent, '%')

    plt.figure(figsize=(9,3))

    plt.subplot(131)
    plt.imshow(img, 'gray')
    plt.title('original image')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(np.log(abs(lossy_dwt)),'gray')
    plt.title('lossy/sparse DWT')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(lossy_idwt, 'gray')
    plt.title('IDFT')
    plt.xticks([])
    plt.yticks([])

    return lossy_idwt

def Image_Sensing_n_Recovery(path, k = 0.2, t = 'DCT', L1 = 0.005):
    '''
    Provide a complete sensing and recovery pipeline for the target image
    '''

    x, w, h = get_img_data(path)
    z, xr = cs.Sensing_n_Recovery(x, k = k, t = t, L1=L1)

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