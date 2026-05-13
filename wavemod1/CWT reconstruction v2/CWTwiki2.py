import numpy as np
import matplotlib.pyplot as plt
import pywt


def morlet_wavelet(t, w=5.0):
    # wavelet = np.exp(2j * np.pi * t) * np.exp(-t ** 2 / (2 * w ** 2))
    wavelet = np.exp(1j * w * t) * np.exp(-0.5 * t ** 2) * np.pi ** (-0.25)
    return wavelet


def cwt(signal, scales, wavelet_function, dt=1.0):
    output = np.zeros((len(scales), len(signal)), dtype=np.complex_)
    for i, scale in enumerate(scales):
        wavelet_data = wavelet_function(np.arange(-len(signal) / 2, len(signal) / 2) * dt / scale) / np.sqrt(scale)
        # plt.figure()
        # plt.plot(wavelet_data)
        # plt.show()
        output[i, :] = np.convolve(signal, wavelet_data, mode='same')
    return output


def icwt(cwt_coefficients, scales, wavelet_function, dt=1.0):
    reconstructed_signal = np.zeros(len(cwt_coefficients[0]))
    for i, scale in enumerate(scales):
        wavelet_data = wavelet_function(
            np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2) * dt / scale) / np.sqrt(scale)
        contribution = np.convolve(cwt_coefficients[i, :], np.conj(wavelet_data[::-1]), mode='same')
        reconstructed_signal += np.real(contribution) / scale

    # Normalization factor
    reconstruction_factor = np.sum((np.abs(wavelet_function(np.linspace(-2, 2, 1000)))) ** 2)
    return reconstructed_signal / reconstruction_factor


def generate_scales(dj, dt, N):
    s0 = 2*dt
    # Largest scale
    J = int((1 / dj) * np.log2(N * dt / s0))
    sj = s0 * 2 ** (dj * np.arange(0, J + 1))
    return sj


t = np.linspace(0, 5, 1000)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 8 * t) * 1

scales = generate_scales(0.125,5/1000,len(signal))
# scales = np.linspace(0.1, 5, 30)

# cwt_coefficients = cwt(signal, scales, morlet_wavelet)
cwt_coefficients, rr = pywt.cwt(signal,scales,wavelet='morl')

reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet)


# Visualize the CWT coefficients
plt.imshow(np.abs(cwt_coefficients), extent=[0, 10, 1, 0.1], aspect='auto', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
plt.show(block = False)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.legend()
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='orange')
plt.legend()
plt.title('Reconstructed Signal')

plt.tight_layout()
plt.show()
