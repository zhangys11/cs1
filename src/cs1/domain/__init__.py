import numpy as np

def frequency_spectrum(x, sr):
    """
    Get frequency spectrum of a signal from time domain. 
    Can be used on audio signal or other 1-D data.

    Parameters
    ----------
    ch : channel signal
    sf : sampling rate

    Return
    ------
    Frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sr)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = np.fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)