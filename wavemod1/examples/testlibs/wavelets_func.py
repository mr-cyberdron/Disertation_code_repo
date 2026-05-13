import numpy as np

def morlet_wavelet(t, scale, omega0 = 5, norm = 1):
    scaled = None
    t = np.arange(0, t) - (t - 1.0) / 2
    t = np.array(t) / scale
    if norm == 1:
        wavelet =  np.pi ** (-0.25) *((np.exp(1j*omega0*t)-np.exp(-(omega0**2/2)))*np.exp(-(t**2/2)))
        scaled = np.sqrt(1/scale) * wavelet
    if norm == 2:
        scaled = np.pi ** (-0.25) * ((np.exp(1j * omega0 * t) - np.exp(-(omega0 ** 2 / 2))) * np.exp(-(t ** 2 / 2)))
    return scaled
