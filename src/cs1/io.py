import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab, matplotlib
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from plotComponents2D import *

def LoadDataset(path, delimiter = ',', has_y = True):
    '''
    Parameters
    ----------
    has_y : boolean, whether there is y column, usually the 1st column.
    '''

    # path = "github/data/754a_C2S_Beimu.txt"
    data = pd.read_csv(path, delimiter=delimiter) # ,header=None

    cols = data.shape[1]

    if has_y:

        # convert from pandas dataframe to numpy matrices
        X = data.iloc[:,1:cols].values # .values[:,::10]
        y = data.iloc[:,0].values.ravel() # first col is y label
        # use map(float, ) to convert the string list to float list
        X_names = list(map(float, data.columns.values[1:])) # list(map(float, data.columns.values[1:])) # X_names = np.array(list(data)[1:])

    else:

        X = data.values
        y = None

        # use map(float, ) to convert the string list to float list
        X_names = list(map(float, data.columns.values)) # X_names = np.array(list(data)[1:])

    return X, y, X_names

def ScatterPlot(X, y):    

    pca = PCA(n_components=2) # keep the first 2 components
    X_pca = pca.fit_transform(X)
    plotComponents2D(X_pca, y)

def Draw_Average (X, X_names):

    matplotlib.rcParams.update({'font.size': 16})

    plt.figure(figsize = (20,5))

    plt.plot(X_names, X.mean(axis = 0), "k", linewidth=1, label= 'averaged waveform $± 3\sigma$ (310 samples)') 
    plt.errorbar(X_names, X.mean(axis = 0), X.std(axis = 0)*3, 
                color = "dodgerblue", linewidth=3, 
                alpha=0.3)  # X.std(axis = 0)
    plt.legend()

    plt.title(u'Averaged Spectrum')
    # plt.xlabel(r'Wavenumber $(cm^{-1})$')
    # plt.ylabel('Raman signal')
    plt.yticks([])
    # plt.xticks([])
    plt.show()

    matplotlib.rcParams.update({'font.size': 12})

def Draw_Class_Average (X, y, X_names):

    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(figsize = (20,5))

    for c in set(y):    
        Xc = X[y == c]
        yc = y[y == c]    
        plt.plot(X_names, np.mean(Xc,axis=0).tolist(), label= 'Brand' + str(c) + ' (' + str(len(yc)) + ' samples)') 
        plt.legend()

    plt.title(u'Averaged Spectrums for Each Category')

    matplotlib.rcParams.update({'font.size': 12})

def PeekDataset(path,  delimiter = ',', has_y = True):

    X, y, X_names = LoadDataset(path, delimiter=delimiter, has_y = True)
    
    if y is None:
        Draw_Average (X, X_names)
    else:
        Draw_Class_Average (X, y, X_names)

    ScatterPlot(X, y)
    
    return X, y, X_names
