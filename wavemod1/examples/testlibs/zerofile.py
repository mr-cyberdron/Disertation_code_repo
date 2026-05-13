from Artef_sig import sigTotest
import pycwt.wavelet as w
import numpy as np
import matplotlib.pyplot as plt

fs = 500
sig = sigTotest(fs=fs, part_t=0.1)
t = np.arange(0, len(sig)/fs, 1/fs)
num_scales = 500
scales = np.arange(1, num_scales)

cwt_res = w.cwt(sig,1/fs, J=num_scales, wavelet='morlet')
cwtmatr = cwt_res[0]
sj = cwt_res[1]
res =  w.icwt(cwtmatr,sj,1/fs,wavelet='morlet')

plt.figure()
plt.subplot(2,1,1)
plt.plot(sig)
plt.subplot(2,1,2)
plt.plot(res)
plt.show()

# Plot the results
plt.figure(figsize=(10,5))
plt.subplot(211)
plt.plot(t, sig)
plt.plot(t,res)
plt.legend(['init_sig','re_sig'])
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.contourf(t, np.arange(1, len(cwtmatr)+1), np.abs(cwtmatr))
plt.colorbar()
plt.title('CWT with Morlet Wavelet')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.show()