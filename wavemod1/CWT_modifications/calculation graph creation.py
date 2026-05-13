import numpy as np
from AnalogFilters import AnalogFilterDesign
from CWT_mod1 import morlet_wavelet,cut_wavelet
import matplotlib.pyplot as plt
def add_noise(input_signal, fs, snr = 130):
    ecg_power = 1
    noise_power = ecg_power / (10 ** (snr / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(input_signal))
    noisy_ecg = input_signal + noise
    # noise_power = np.mean(noise**2)
    # snr = 10 * np.log10(ecg_power / noise_power)
    # input(snr)

    # signal_t = np.array(list(range(len(noise)))) / 1000
    # ax1 = plt.subplot(2, 1, 1)
    # plt.plot(signal_t, noise)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')
    # ax1.legend(['Чистий ЕКГ', 'ППП' ,'ППШ', 'Нерегулярна складова', 'Шум'])

    return noisy_ecg



fs = 1000
signal = np.zeros(int(fs/2))
signal2 = np.zeros(int(fs/2))
t_wavelet = np.arange(-len(signal) / 2, len(signal) / 2)

noise = add_noise(signal,fs,15)
noise2 = add_noise(signal2,fs,5)
noise_to_check = AnalogFilterDesign(noise, fs).lp(order=5, cutoff=80).zerophaze().butter() \
    .filtration()
noise_to_check2 = AnalogFilterDesign(noise2, fs).lp(order=5, cutoff=10).zerophaze().butter() \
    .filtration()

morl = morlet_wavelet(t_wavelet/15)
morl_cut = cut_wavelet(morl)


print(f'Calculated wavelet_len {(6*(15))/(6*1000)}')
print(f'wavelet_len {len(morl_cut)}')


def shifted_wave(morl_cut, sig, start = 30):
    morl2 = np.empty(len(sig))
    morl2[:] = np.nan
    morl2 = list(morl2)
    morl_cut = list(morl_cut)
    morl2 = morl2[:start] + morl_cut + morl2[start-len(morl_cut):]
    return morl2

background = noise_to_check+noise_to_check2+morl
morl1 = morl_cut[int(round((len(morl_cut)/2)-1)):]
morl2 = shifted_wave(morl_cut,noise_to_check,start = 75)
morl3 = shifted_wave(morl_cut,noise_to_check,start = 199)
morl4 = shifted_wave(morl_cut,noise_to_check,start = 320)
morl5_1 = morl_cut[0:int(round((len(morl_cut)/2)))]
morl5 = shifted_wave(morl5_1,noise_to_check,start = 448)

plt.figure()
plt.plot(background, color = 'black')
plt.plot(morl1,linestyle = ':', color='red', linewidth = 1.5)
plt.plot(morl2,linestyle = ':', color='red',linewidth = 1.5)
plt.plot(morl3,linestyle = ':', color='red',linewidth = 1.5)
plt.plot(morl4,linestyle = ':', color='red',linewidth = 1.5)
plt.plot(morl5,linestyle = ':', color='red',linewidth = 1.5)
plt.show(block = False)

plt.figure(figsize=(15, 1))
data = np.array([0.1, 0.095 , 0.015 ,0.40, 0.21,0.35 ,0.9, 0.25 , 0.02,0.3, 0.16,0.1 ,0.35])

# Convert the 1D array to a 2D array with 1 row
data_2d = data[np.newaxis, :]

# Plot using imshow
plt.imshow(data_2d, aspect='auto', cmap='viridis')  # 'aspect' can be set to 'auto' to stretch the cells

# Adding a color bar to indicate the scale
plt.colorbar(label='Value scale')

# Remove y-axis since there's only one row
plt.yticks([])

plt.show(block = False)

plt.figure()
noise_to_check3 = AnalogFilterDesign(noise2, fs).lp(order=5, cutoff=5).zerophaze().butter() \
    .filtration()
noise_to_check4 = AnalogFilterDesign(noise, fs).lp(order=5, cutoff=50).zerophaze().butter() \
    .filtration()
walked_wavelet = noise_to_check3*10+noise_to_check4+morl
walked_wavelet = walked_wavelet[150:350]
mean = np.mean(walked_wavelet)*np.ones(len(walked_wavelet))
corrected_wwavelet = np.array(walked_wavelet)-mean

plt.plot(walked_wavelet, c = 'black')
plt.plot(np.zeros(len(walked_wavelet)), c = 'black', linestyle = '-.')
morl_toplot = shifted_wave(morl_cut,morl[150:350],48)
plt.plot(morl_toplot, c = 'red', linestyle = ':',linewidth = 1.72)
plt.plot(mean, c = 'orange',linewidth = 1)
plt.show(block = False)


plt.figure()
plt.plot(np.array(walked_wavelet) - mean, c = 'black')
plt.plot(np.zeros(len(walked_wavelet)), c = 'black', linestyle = '-.')
morl_toplot = shifted_wave(morl_cut,morl[150:350],48)
plt.plot(morl_toplot, c = 'red', linestyle = ':',linewidth = 1.72)
plt.plot(np.zeros(len(np.array(walked_wavelet) - mean)), c = 'orange',linewidth = 1)
plt.show()



