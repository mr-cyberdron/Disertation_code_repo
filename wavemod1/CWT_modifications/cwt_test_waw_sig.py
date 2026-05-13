import librosa
import matplotlib.pyplot as plt
import numpy as np
from CWT_mod1 import cwt,icwt,morlet_wavelet,scale_to_frequency
import wave
import numpy as np



def scale_to_frequency(scales, w0, fs):
    frequencies = (w0 / (2 * np.pi * scales)) * fs
    return frequencies
print('loading')
y, sr = librosa.load('vestibyul-kontsertnogo-zala-vecherinka-tolpa-govoryaschih-lyudey-rus-24981.mp3')
print(f'fs{sr}')

scales = np.linspace(5,40,50)


y = y[0:int(np.round(len(y)/8))]
# y = y[30000:40000]
cwt_coefficients = cwt(y, scales, morlet_wavelet, dt=1, fs=sr,w0=6, plot_wavelets_spectrum=False,Amp_correction_target_amp=0.06)
reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet, dt=1, ds=1)*300*32000

# plt.figure()
# plt.plot(reconstructed_signal)
# plt.show()

file = wave.open('test.wav', 'w')
file.setnchannels(1)  # моно
file.setsampwidth(2)  # размер выборки в байтах
file.setframerate(sr)
file.writeframes(reconstructed_signal.astype(np.int16).tobytes())
file.close()

# plt.show()

