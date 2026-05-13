import pywt
import numpy as np
from Artef_sig import doublesin
import matplotlib.pyplot as plt
from Artef_sig import sigTotest


fs = 500
sig = sigTotest(fs=fs,part_t=0.1)
t = np.arange(0, len(sig)/fs, 1/fs)
wavelet = 'morl'
num_scales = 80
scales = np.arange(1, num_scales)

# Perform CWT using PyWavelets
cwtmatr, freqs = pywt.cwt(sig, scales,wavelet, method='conv')

def icwt(coefficients, scales):
    c1 = -1
    c2 = np.sqrt(scales)
    c3 = np.real(coefficients)
    return c1*(c3/np.transpose([c2])).sum(axis=0)

recsig =   icwt(cwtmatr,scales)

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
