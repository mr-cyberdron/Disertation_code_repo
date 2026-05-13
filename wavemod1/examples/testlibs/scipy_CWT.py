import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from Artef_sig import sigTotest

fs = 500
sig = sigTotest(fs=fs,part_t=0.1)
t = np.arange(0, len(sig)/fs, 1/fs)
num_scales = 80
scales = np.arange(1, num_scales)
wavelet = signal.morlet2

cwtmatr = signal.cwt(sig, wavelet, scales, dtype=np.float64)

#cwtmatr_yflip = np.flipud(cwtmatr)

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


