import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv, svd, norm

from .cs import *

def LDA(X, y, display = True):
    '''
    Linear Discriminant Analysis

    Return
    ------
    U, S, V = SVD(SS) = SVD(Sb/Sw)
    PSI = Uh
    Z = X@U
    Xr = Z@PSI
    '''
    m,n = X.shape
    labels = set(y)

    class_feature_means = {}
    feature_means = np.mean(X, axis = 0)
    within_class_scatter_matrix = np.zeros((n,n))
    between_class_scatter_matrix = np.zeros((n,n))

    for c in labels:    
        Xc = X[y == c]
        yc = y[y == c]
        class_feature_means[c] = np.mean(Xc,axis=0)    
        mc = class_feature_means[c]
        
        within_class_scatter_matrix += (Xc-mc).T @ (Xc-mc)
        demean = (mc - feature_means).reshape(-1,1)
        between_class_scatter_matrix += len(yc) * demean @ demean.T / len(y)

    # Solve the generalized eigenvalue problem for 𝑆𝑆=𝑆𝑏/𝑆𝑤 to obtain the linear discriminants.
    SS = pinv(within_class_scatter_matrix).dot(between_class_scatter_matrix)

    if display:

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        ax[0].imshow(within_class_scatter_matrix, cmap = 'gray')
        ax[0].axis('off')
        ax[0].set_title('within-class scatter matrix (Sw)')

        ax[1].imshow(between_class_scatter_matrix, cmap = 'gray')
        ax[1].axis('off')
        ax[1].set_title('between-class scatter matrix (Sb)')

        ax[2].imshow(SS, cmap = 'gray')
        ax[2].axis('off')
        ax[2].set_title('SS = Sb/Sw')

        fig.tight_layout()
        plt.show()

    U,S,Vh = svd(SS)
    PSI = U.conj().T # or just U.T. for real-value matrix, conj equals itself

    Z = X @ U
    # plotComponents2D(Z, y, set(y))
    Xr = Z @ U.T

    if display:

        plt.figure(figsize=(9,3))
        plt.plot(X[0], color='gray')
        plt.title('X[0] - 1st sample')
        plt.show()

        plt.figure(figsize=(9,3))
        plt.plot(Z[0], color='gray')
        plt.title('Z[0] - 1st sample in latent space')
        plt.show()

        plt.figure(figsize=(9,3))
        plt.plot(Xr[0])
        plt.title('Xr[0] - 1st sample recovered')
        plt.show()

    return PSI, Z, Xr

def LDA_Sensing_n_Recovery(PSI, x, k = 0.2):

    PHI, OMEGA = GetSensingMatrix(len(x), k)

    A = PHI @ PSI
    xs = PHI @ x.T

    lasso = Lasso(alpha=0.001, tol = 0.005) # The default for tol is 0.0001. Increase tol to prevent Convergence Warning.
    lasso.fit(A, xs)
    print ('non-zero coefs: ', np.count_nonzero(lasso.coef_))

    xr = PSI @ lasso.coef_ 

    plt.figure(figsize=(9,3))
    plt.plot(x.T, color='gray')
    plt.title('original signal')
    plt.show()

    plt.figure(figsize=(9,3))
    plt.plot(xr.T, color='gray')
    plt.title('recovered signal')
    plt.show()


def EBP(X, display = True):
    '''
    Eigenvector-Based Projection via SVD (Singular Value Decomposition)

    Return
    ------
    Q, R, Wh = SVD(X)
    '''
    Q, R, Wh = svd(X)
    # Wh is already transposed. WH = W.T  
    # W is the PSI for PCA  # W.conj().T == inv(W)
    # Each column is an eigenvector, or PC loadings
    W = Wh.conj().T    

    r = np.linalg.matrix_rank(X) 
    # plt.imshow(W) # After the r-th element, almost an identity matrix, 
    # i.e. diagnonal element is 1, others are 0.

    #
    # We can also check that the first orthogonal matrix from SVD of Cov equals the third matrix from SVD of X (W).
    CM = np.cov(X, rowvar = False)
    Q2, _, _= svd(CM) 

    if display:

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        ax[0].matshow(W[:r,:r], cmap = 'gray')
        ax[0].axis('off')
        ax[0].set_title('3rd SVD rotational component of data matrix\nShow the upper-left r sub-matrix. \nr is the rank of X. r = ' + str(r))

        ax[1].matshow(Q2[:r,:r], cmap = 'gray') # Begin from the 31th element, almost an identity matrix, i.e. diagnonal element is 1, others are 0.
        ax[1].set_title('1st SVD rotational component of cov matrix\nShow the upper-lef r sub-matrix. \nr is the rank of X. r = ' + str(r))
        ax[1].axis('off')

        fig.tight_layout()
        plt.show()

    return W, Q2

def EBP_Sensing_n_Recovery(PSI, x, n_components = None, k = 0.2):

    if n_components is None:
        n_components = len(x)

    PHI, OMEGA = GetSensingMatrix(len(x), k)

    A = PHI @ PSI[:,:n_components]
    xs = PHI @ x.T

    lasso = Lasso(alpha=0.001, tol = 0.005) # The default for tol is 0.0001. Increase tol to prevent Convergence Warning.
    lasso.fit(A, xs)
    print ('non-zero coefs: ', np.count_nonzero(lasso.coef_))

    xr = PSI[:,:n_components] @ lasso.coef_ 

    plt.figure(figsize=(9,3))
    plt.plot(x.T, color='gray')
    plt.title('original signal')
    plt.show()

    plt.figure(figsize=(9,3))
    plt.plot(xr.T, color='gray')
    plt.title('recovered signal')
    plt.show()