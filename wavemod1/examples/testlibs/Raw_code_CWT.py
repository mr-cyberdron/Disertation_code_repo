import numpy as np
import matplotlib.pyplot as plt
from wavelets_func import morlet_wavelet
from Artef_sig import sigTotest
from scipy import signal as s
import pywt

import pycwt.wavelet as w

def convolve(signal1, signal2, mode = None):
    print('conv_calc')
    assert len(signal1) == len(signal2)
    # Get the dimensions of the signals
    signal1_length = len(signal1)
    signal2_length = len(signal2)

    # Create an empty output signal
    output_signal = [0] * (signal1_length + signal2_length - 1)

    # Iterate over the output signal and apply the convolution
    for i in range(signal1_length + signal2_length - 1):
        for j in range(max(0, i - signal2_length + 1), min(i + 1, signal1_length)):
            output_signal[i] += signal1[j] * signal2[i - j]

    if mode == 'same':
        fromm = int(np.round(signal1_length/2)-1)
        to = int(signal1_length+fromm)
        output_signal = np.array(output_signal)[fromm:to]

    return output_signal

def cwt(signal, scales, complex = False):
    n = len(signal)
    cwt_matrix = np.zeros((len(scales), n),dtype='complex_')
    for ind, scale in enumerate(scales):
        wavelet_data = np.conj(morlet_wavelet(n, scale))
        # wavelet_data = s.morlet2(n,scale,5)
        cwt_matrix[ind, :] = np.convolve(signal, wavelet_data, mode='same')
        #cwt_matrix[ind, :] = convolve(signal, wavelet_data, mode='same')
        if complex:
            cwt_matrix = cwt_matrix
        else:
            cwt_matrix = np.real(cwt_matrix)
    return cwt_matrix


def icwt(coefficients, scales):
    c1 = -1
    c2 = np.sqrt(scales)
    c3 = np.real(coefficients)
    return c1*(c3/np.transpose([c2])).sum(axis=0)


fs = 500
sig = sigTotest(fs=fs, part_t=0.1)
t = np.arange(0, len(sig)/fs, 1/fs)
num_scales = 80

omega0 = 5
morlet_flambda = (4 * np.pi) / (omega0 + np.sqrt(2 + omega0**2))
s0 = 2 * (1/fs) / morlet_flambda
dj = 1/12

sj = s0 * 2 ** (np.arange(0, num_scales + 1) * dj)
scales = sj



cwtmatr = cwt(sig, scales, complex=True)

recsig = icwt(cwtmatr,scales)

plt.figure()
plt.subplot(2,1,1)
plt.plot(sig)
plt.subplot(2,1,2)
plt.plot(recsig)
plt.legend(['init', 'rec_sig'])
plt.show()



# Plot the results
plt.figure(figsize=(10,5))
plt.subplot(211)
plt.plot(t, sig)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.contourf(t, scales, np.abs(cwtmatr))
plt.colorbar()
plt.title('CWT with Morlet Wavelet')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.show()

