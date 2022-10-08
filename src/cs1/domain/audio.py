from pydub import AudioSegment, playback
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.signal import wavelets
from scipy.io import wavfile
import os


def simulate_ECG(bpm = 60, time_length = 10, display = True):
    '''
    Parameters
    ----------
    bpm : Simulated Beats per minute rate. For a health, athletic, person, 60 is resting, 180 is intensive exercising
    time_length : Simumated length of time in seconds

    Return
    ------
    ecg : a simulated ECG signal
    '''

    bps = bpm / 60
    capture_length = time_length

    # Caculate the number of beats in capture time period 
    # Round the number to simplify things
    num_heart_beats = int(capture_length * bps)


    # The "Daubechies" wavelet is a rough approximation to a real,
    # single, heart beat ("pqrst") signal
    pqrst = wavelets.daub(10)

    # Add the gap after the pqrst when the heart is resting. 
    samples_rest = 10
    zero_array = np.zeros(samples_rest, dtype=float)
    pqrst_full = np.concatenate([pqrst,zero_array])


    # Concatonate together the number of heart beats needed
    ecg_template = np.tile(pqrst_full , num_heart_beats)

    # Add random (gaussian distributed) noise 
    noise = np.random.normal(0, 0.01, len(ecg_template))
    ecg = noise + ecg_template


    if display:

        # Plot the noisy heart ECG template
        plt.figure(figsize=(int(time_length * 2), 3))
        plt.plot(ecg)
        plt.xlabel('Sample number')
        plt.ylabel('Amplitude (normalised)')
        plt.title('Heart ECG Template with Gaussian noise')
        plt.show()
        
    return ecg

def dct_lossy_signal_compression(x, percent = 99):    
    '''
    DCT lossy compression and recovery for 1D signal
    '''

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
    '''
    DFT lossy compression and recovery for 1D signal
    '''

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


def play_wav(path):
    '''
    Play a wave file
    '''

    if (not os.path.isfile(path)):
        print(path, 'is not a valid file path')
        return

    song = AudioSegment.from_wav(path)
    playback.play(song) # need to install ffmpg

def read_wav(path, ch=None, display=True):
    '''
    Parse a wave file and get its data

    Parameters
    ----------
    ch : channel. If unspecified, will return all channels' data

    Return 
    ------
    data : the specified channel data (if ch is not None); otherwise, all channels' data
    rate : sampling rate
    '''

    rate, data = wavfile.read(path)
    print('data shape: ', data.shape)
    print("channels:", data.shape[1])    
    length = data.shape[0] / rate
    print('length:',length)
    print('sampling rate:', rate)
    
    if display == True:
        OFFSET = data.max()*1.5
        time = np.linspace(0., length, data.shape[0])
        plt.figure(figsize = (12,6))
        for c in range(data.shape[1]):
            plt.plot(time, data[:, c] + OFFSET*c, label="channel "+str(c), alpha=0.3)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    if not ch:
        return data, rate
        
    if not (ch >= 0 and ch < data.shape[1]):
        raise ValueError('ch is invalid. should be an integer within [0,nchannels)')
    
    return data[:, ch], rate

def audio_frequency_spectrum(x, sr):
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

def analyze_audio_signal(ch, sr, frange = 1.0):
    """
    Analyze the specified channel signal.

    Parameters
    ----------
    ch : channel signal
    sr : sampling frequency
    """

    y = ch  # use the first channel (or take their average, alternatively)
    print(y.shape)
    t = np.arange(len(y)) / float(sr)

    plt.figure(figsize=(12,12))
    plt.subplot(3, 1, 1)
    plt.plot(t, y, alpha=0.3)
    plt.xlabel('time (s)')
    plt.ylabel('signal')

    frq, X = audio_frequency_spectrum(y, sr)

    plt.subplot(3, 1, 2)
    ub = math.floor(len(frq)*frange)
    plt.plot(frq[:ub], X[:ub], alpha=0.3)
    plt.xlabel('freq (Hz)')
    plt.ylabel('|FFT|')

    plt.subplot(3, 1, 3)
    fourier = np.fft.fft(y)
    FFT = abs(fourier)
    freq = np.fft.fftfreq(len(y), d=1/sr) # d - Sample spacing (inverse of the sampling rate). Defaults to 1.
    plt.plot(freq, FFT, alpha=0.3)
    plt.xlabel('freq (Hz)')
    plt.ylabel('|FFT| two-sided')

    plt.tight_layout()

    plt.show()

def analyze_wav(path, frange = 1.0):
    """
    :param path: wave file path
    :param frange: the first frange percentage to be shown in the spectrum plot
    """
    if (not os.path.isfile(path)):
        print(path, 'is not a valid file path')
        return
    
    data, rate = read_wav(path)

    for i in range(data.shape[1]):
        print('--- CH' + str(i) + ' ---')
        analyze_audio_signal(data[:, i], rate, frange)
        
    return data, rate

def save_wav(path, y, rate = 48000, play = False):
    '''
    Save data as a wave file

    Parameters
    ----------
    play : whether play the sound file after saving
    '''

    wavfile.write(path, rate=rate, data = y.astype(np.int16))
    
    if play:
        play_wav(path)

def split_wav(path, cut_length = 1):
    '''
    Split a wav file to n pieces.

    path : input wav file path
    cut_length : time length in second
    '''

    f = wave.open(path, 'rb')
    params = f.getparams() #读取音频文件信息
    nchannels, sampwidth, framerate, nframes = params[:4]  #声道数, 量化位数, 采样频率, 采样点数   
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    #根据声道数对音频进行转换
    if nchannels > 1:
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        temp_data = wave_data.T
    else:
        wave_data = wave_data.T
        temp_data = wave_data.T

    CutFrameNum = framerate * cut_length  
    Cutnum =nframes/CutFrameNum  #音频片段数
    StepNum = int(CutFrameNum)
    StepTotalNum = 0

    for j in range(int(Cutnum)):
        FileName = path + "-" + str(j) + ".wav" 
        temp_dataTemp = temp_data[StepNum * (j):StepNum * (j + 1)]
        StepTotalNum = (j + 1) * StepNum
        temp_dataTemp.shape = 1, -1
        temp_dataTemp = temp_dataTemp.astype(np.short)# 打开WAV文档
        f = wave.open(FileName, 'wb')
        # 配置声道数、量化位数和取样频率
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(temp_dataTemp.tostring())  # 将wav_data转换为二进制数据写入文件
        f.close()
        
