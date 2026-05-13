from wavelets import WaveletAnalysis
import numpy as np
import matplotlib.pyplot as plt

#pip install git+https://github.com/endolith/wavelets

fs = 1000
t_sec = 10
n_samps = fs*t_sec
t = np.linspace(0, t_sec, n_samps)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 8 * t) * 1

dt = t_sec/n_samps

wa = WaveletAnalysis(signal, dt=dt)

# # wavelet power spectrum
coefs = wa.wavelet_transform

# scales
scales = wa.scales


# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()

plt.figure()
plt.subplot(3,1,1)
plt.plot(signal)
plt.subplot(3,1,2)
plt.plot(rx)
plt.subplot(3,1,3)
plt.imshow(np.abs(coefs), extent=[t[0], t[-1], scales[0], scales[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
plt.show()