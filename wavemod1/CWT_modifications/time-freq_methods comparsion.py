import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from tftb.processing import WignerVilleDistribution
from CWT_modifications.Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
from CWT_modifications.Artifitial_signal_creation import sigTotest
import matplotlib.pyplot as plt
from CWT_basis import cwt,icwt,morlet_wavelet,scale_to_frequency
import numpy as np
import matplotlib
import copy

w0 = 6
fs = 1000
sig = simulate_ecg_with_VLP_ALP(duration = 4, #sec
                              fs = fs, #hz
                              noise_level =130, #db
                              hr = 80,#bpm
                              Std = 0, #bpm
                              unregular_comp = False,
                                random_state = 11,
                              lap_amp = 20,
                              lvp_amp = 40)

sig = sig[500:1200]

t_vector = np.array(list(range(len(sig)))) / fs
scales = np.linspace(3, 200, 200)
cwt_coefficients = cwt(sig, scales, morlet_wavelet, dt=1, fs=fs, w0=6, plot_wavelets_spectrum=False)

f_stft, t_stft, Zxx = stft(sig, fs,nperseg=100,noverlap=90, return_onesided=False)

# shifting the frequency axis for better representation
Zxx = np.fft.fftshift(Zxx, axes=0)
f_stft = np.fft.fftshift(f_stft)

# Doing the WVT
wvd = WignerVilleDistribution(sig, timestamps=t_vector)
tfr_wvd, t_wvd, f_wvd = wvd.run()


ts = t_vector
dt = 1/fs

f, axx = plt.subplots(4, 1)


axx[0].plot(t_vector,sig)
axx[0].set_ylabel('[mV]')

start_time = 0.16
end_time = 0.19
start_time2 = 0.31
end_time2 = 0.34

red_part = copy.deepcopy(sig)
indices_to_keep = np.r_[np.round(start_time*fs).astype(int):np.round(end_time*fs).astype(int),
                  np.round(start_time2*fs).astype(int):np.round(end_time2*fs).astype(int)]  # объединяем диапазоны, Python использует включительные диапазоны для срезов
red_part[~np.in1d(np.arange(red_part.size), indices_to_keep)] = np.nan  # заменяем все, что вне диапазонов

axx[0].plot(t_vector,red_part, c='r')

axx[0].grid()

df1 = f_stft[1] - f_stft[0]  # the frequency step
im = axx[1].imshow(np.log(np.real(Zxx * np.conj(Zxx))), aspect='auto',
          interpolation=None, origin='lower',
          extent=(ts[0] - dt/2, ts[-1] + dt/2,
                  f_stft[0] - df1/2, f_stft[-1] + df1/2))
axx[1].set_ylabel('[Hz]')
axx[1].set_ylim(0,200)
plt.colorbar(im, ax=axx[1])
plt.colorbar(im, ax=axx[0])
# axx[1].set_title('spectrogram')


# because of how they implemented WVT, the maximum frequency is half of
# the sampling Nyquist frequency, so 125 Hz instead of 250 Hz, and the sampling
# is 2 * dt instead of dt
f_wvd = np.fft.fftshift(np.fft.fftfreq(tfr_wvd.shape[0], d=2 * dt))
df_wvd = f_wvd[1]-f_wvd[0]  # the frequency step in the WVT
im = axx[2].imshow(np.log(np.fft.fftshift(tfr_wvd, axes=0)+35), aspect='auto', origin='lower',
       extent=(ts[0] - dt/2, ts[-1] + dt/2,
               f_wvd[0]-df_wvd/2, f_wvd[-1]+df_wvd/2))
axx[2].set_xlabel('time [s]')
axx[2].set_ylabel('[Hz]')
axx[2].set_ylim(0,200)
plt.colorbar(im, ax=axx[2])
# axx[2].set_title('Wigner-Ville Transform')


freqs = scale_to_frequency(scales, w0, fs)
X, Y = np.meshgrid(t_vector, freqs)
cwt_img = axx[3].pcolormesh(X, Y, np.log(np.abs(cwt_coefficients)), cmap='viridis')
plt.colorbar(cwt_img, ax=axx[3])
axx[3].set_xlabel('time [s]')
axx[3].set_ylabel('[Hz]')
axx[2].set_ylim(0,200)
# axx[3].set_title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
plt.show()

