'''
Contains CS basic operations and pipelines
'''
import os, sys
import numpy as np
import pandas as pd
import scipy
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
import pywt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Lasso, LassoCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from statsmodels.multivariate.manova import MANOVA

if __package__:
    from . import GetSensingMatrix, SensingWithPHI, PSI_LONGNAMES, PSI_NAMES
    from .basis.common import Generate_PSI
    from .metrics import Univariate_KLD, Multivarate_KLD, calculate_recon_error
else:
    DIR = os.path.dirname(__file__)  # cs1 dir
    if DIR not in sys.path:
        sys.path.insert(0,DIR)
    from __init__ import GetSensingMatrix, SensingWithPHI, PSI_LONGNAMES, PSI_NAMES
    from basis.common import Generate_PSI
    from metrics import Univariate_KLD, Multivarate_KLD, calculate_recon_error


def Sensing(x, k = 0.2, flavor = 'identity'):      
    '''
    信号采样，发送端处理

    Parameters
    ----------
    k : (0,1]. sampling ratio.
    flavor : 
        'identity' / 'bernoulli' - use a scramble identity matrix. take its n*k rows as sensing matrix
        'gaussian' - sampling from N(1, 1/n)

    Return
    ------
    xs : sampled signal
    r : sampling indices
    '''

    n = len(x)
    
    if flavor == 'gaussian':
        PHI,_ = GetSensingMatrix(n, k = k, flavor = flavor)
        return SensingWithPHI(x, PHI), PHI
    
    pm = np.random.permutation(np.arange(1, n))
    r = pm[:int(n*k)]
    xs = x[r]

    return xs, r

# from scipy.sparse import csr_matrix
def MeasurementMatrix (N, r, t = 'DCT'): 

    '''
    Construct the measurement matrix. 
    Currently only support DCT and DFT.
    '''
    
    kn = len(r)    
    a = np.zeros((1, N))  
    t = t.upper()
        
    # Suppose the first number in the permutation is 42. Then choose the 42th
    # elements in a matrix full of zeros and change it to 1. Apply a discrete
    # cosine transformation to this matrix, and add the result to other matrix
    # Adelta as a new row.
    # A = np.zeros((kn, N))

    A = np.zeros ((kn, N)) #csr_matrix, dtype='float16') # reduce memory use. numpy uses float64 as default dtype.

    for i, j in enumerate(r):
        a[0,j] = 1
        if t == 'IDM' or t == 0:
            A[i,:] = a # .astype(np.float16)
        elif t == 'DCT' or t == 1:
            A[i,:] = cv2.dct(a)# .flatten() # .astype(np.float16)
        elif t == 'DFT' or t == 2:
            A[i,:] = cv2.dft(a)# .flatten() # .astype(np.float16)
        a[0,j] = 0
    
    ###### ALTERNATIVE IMPLEMENTATION ######
    # phi = np.zeros((k, N)) # Sampling matrix, kxN    
    # for i, j in enumerate(r):
    #    phi[i,j] = 1
    # psi = dctmtx(N,N) # memory costly
    # A = phi*psi
    
    return A # .toarray()

def Recovery (A, xs, t = 'DCT', PSI = None, solver = 'LASSO', L1 = 0.005, \
    display = True, verbose = False):
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
    solver : 'LASSO' or 'OMP' or 'VAE' (variational autoencoder, need a pretrained VAE model)
    L1 : LASSO L1. specify a float or leave None for CV.
    '''

    t = t.upper()

    if solver == 'LASSO':

        alphas = None # automatic set alphas
        if  t == PSI_NAMES[5] or t == PSI_LONGNAMES[5] or \
            t == 'DFT' or \
            t == 'DWT' or \
            t == 'LDA':
            alphas = [0.005, 0.0001, 0.00001] # set empirical alpha values. LDA needs smaller alpha

        if isinstance(L1, float):
            lasso = Lasso(alpha=L1, tol = 0.0005)    
        else:
            lasso = LassoCV(alphas = alphas, selection = 'random', tol = 0.001, n_jobs=-1, verbose = verbose) # ‘random’ often leads to significantly faster convergence especially when tol is higher than 1e-4.
            # 实测并未发现 selction = cyclic / random 两种模式的运行时间差异
        
        lasso.fit(A, xs)
        z = lasso.coef_

    else: # 'OMP'

        # plot the noise-free reconstruction
        omp = OrthogonalMatchingPursuit() # n_nonzero_coefs default 10%.
        omp.fit(A, xs)
        z = omp.coef_
        # print('OMP score:', omp.score(A, xs)) # Return the coefficient of determination R^2 of the prediction.
    
    if t == PSI_NAMES[0] or t == PSI_LONGNAMES[0]:
        z = np.linalg.pinv(A) @ xs
        xr = z
    elif  t == PSI_NAMES[1] or t == PSI_LONGNAMES[1]:
        xr = cv2.idct(z)
    elif t == PSI_NAMES[2] or t == PSI_LONGNAMES[2]:
        xr = cv2.idft(z)
    elif t == PSI_NAMES[3] or t == PSI_LONGNAMES[3] or \
        t == PSI_NAMES[4] or t == PSI_LONGNAMES[4] or \
        t == PSI_NAMES[5] or t == PSI_LONGNAMES[5] or \
        t == 'EBP' or t == 'LDA' : # these are adaptive bases
        xr = (PSI @ z).T
    else:
        print ('unsupported transform type: ', t)
        return

    z = z.flatten()
    xr = xr.flatten()

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

def Sensing_n_Recovery(x, k = 0.2, t = 'DCT', solver = 'LASSO', \
    L1 = 0.005, display = True):
    
    '''
    Provide a complete pipeline for signal sensing and recovery with a speicfic PSI. 

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

    z,xr = Recovery (A, xs, t = t, solver = solver, L1=L1, display = False)

    if display:

        matplotlib.rcParams.update({'font.size': 20})

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32,6))

        if np.any(np.iscomplex(z)):
            ax[0].plot(np.abs(z), color='gray')
        else:
            ax[0].plot(z, color = 'gray')

        ax[0].set_title('recovered latent representation (z)')

        ax[1].plot(xr, color = 'gray')
        ax[1].set_title('reconstructed signal (xr)')

        fig.tight_layout()
        plt.show()

        matplotlib.rcParams.update({'font.size': 12})

    return z, xr

def GridSearch_Sensing_n_Recovery(x, PSIs, ks = [0.1, 0.2, 0.5, 1.001], solver = 'LASSO'):
    '''
    Provide a complete pipeline for signal sensing and recovery with a set of candidate PSIs and hyper-parameters.
    Use grid search strategy to find the best.

    Parameters
    ----------
    x : a single data sample (a vector) 
    '''

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
        
        # PSI_H = PSI.conj().T
        # PSI_INV = np.linalg.pinv(PSI)    
        # assert np.allclose(PSI_H, PSI_INV) # Unitary Matrix        
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

        if np.any(np.iscomplex(z)):
            plt.plot(np.abs(z), color='gray')
        else:
            plt.plot(z, color = 'gray')
                
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


def Dataset_Sensing_n_Recovery (X, y = None, k = 0.2, t = 'DCT', solver = 'LASSO', L1 = 0.005, display = 'all'):
    '''
    Provide a complete pipeline for an entire dataset. 
    PCA 2D visualization and multivariate KLD are evaluated.

    Parameters
    ----------
    X, y : target dataset and labels
    display : 'all' - display all samples
              'first' - display only the first sample
              'none' - don't display
    '''

    def plotComponents2D(X, y, labels = None, use_markers = False, ax=None, legends = None, tags = None):
        '''
        This is a copy of qsi.vis.plotComponents2D()
        '''

        if X.shape[1] < 2:
            print('ERROR: X MUST HAVE AT LEAST 2 FEATURES/COLUMNS! SKIPPING plotComponents2D().')
            return
        
        # Gray shades can be given as a string encoding a float in the 0-1 range
        colors = ['0.9', '0.1', 'red', 'blue', 'black','orange','green','cyan','purple','gray']
        markers = ['o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H']

        if (ax is None):
            fig, ax = plt.subplots()
            
        if (y is None or len(y) == 0):
            labels = [0] # only one class
        if (labels is None):
            labels = set(y)

        i=0        

        for label in labels:
            if y is None or len(y) == 0:
                cluster = X
            else:
                cluster = X[np.where(y == label)]
            # print(cluster.shape)

            if use_markers:
                ax.scatter([cluster[:,0]], [cluster[:,1]], 
                        s=40, 
                        marker=markers[i], 
                        facecolors='none', 
                        edgecolors=colors[i+3],
                        label= (str(legends[i]) if legends is not None else ("Y = " + str(label)  + ' (' + str(len(cluster)) + ')')) )
            else:
                ax.scatter([cluster[:,0]], [cluster[:,1]], 
                        s=70, 
                        facecolors=colors[i],  
                        label= (str(legends[i]) if legends is not None else ("Y = " + str(label) + ' (' + str(len(cluster)) + ')')), 
                        edgecolors = 'black', 
                        alpha = .4) # cmap='tab20'                
            i=i+1
        
        if (tags is not None):
            for j,tag in enumerate(tags):
                ax.annotate(str(tag), (X[j,0] + 0.1, X[j,1] - 0.1))
            
        ax.legend()

        ax.axes.xaxis.set_visible(False) 
        ax.axes.yaxis.set_visible(False)
        
        return ax

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