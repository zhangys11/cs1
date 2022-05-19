# cs1

Compressed Sensing library for 1D Spectroscopic Profiling

# Installation

> pip install cs1

# A simple startup
    
    from cs1.cs import *

    # Generate common non-adaptive bases and save to a local pickle file.
    # The generation process can be very slow, so save it for future use.
    cs.Generate_PSIHs(n, savepath = 'PSIHs_' + str(n) + '.pkl') # n is the data/signal dimensionality

    # load back bases
    file = open('PSIs_' + str(n) + '.pkl','rb')
    PSIs = pickle.load(file)
    file.close()

    # sparsity analysis
    Analyze_Sparsity(x, PSIs)

<img src='sparsity_analysis.png'>

    # compare different bases and sampling ratio on a single sample
    rmses = GridSearch_Sensing_n_Recovery(x, PSIs, solver = 'LASSO') # returns relative MSEs

<img src='grid_search.png'>


# low-level cs functions
    
    dftmtx()
    dctmtx()
    hwtmtx()
    Sensing()
    Recovery()
    Mutual_Coherence()
    ...

# singal processing functions for other domains

    Simulate_ECG()
    dct_lossy_signal_compression()
    dft_lossy_signal_compression()
    img_dct()
    img_dft()
    dct_lossy_image_compression()
    dft_lossy_image_compression()


# adaptive cs bases

    from cs1.adaptive import *
    
    PSI, _ = EBP(X) # X is a m-by-n training dataset. PSI is the EBP basis
    PSI, _, _ = LDA(X, y, display = True) # X and y are training dataset. PSI is the LDA basis.

