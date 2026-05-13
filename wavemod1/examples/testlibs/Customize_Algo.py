import numpy as np
import matplotlib.pyplot as plt


from Artef_sig import sigTotest
from wavelets_func import morlet_wavelet

import SimilarityFunctionsDevelop as sf

from SF_test import compare_SF

import pywt

from scipy import signal

import pandas as pd



def convolve_cosine_sim_based_mod(sig, wavelet):
    if len(wavelet)%2 != 0:
        wavelet = wavelet[1:]
    wavle_half_pos = int(np.round(len(wavelet)/2))
    padded_sig = np.pad(sig, (wavle_half_pos, wavle_half_pos), 'constant', constant_values=(0, 0))


    similarity_results_mass = np.zeros(len(padded_sig))
    for i in range(len(padded_sig)):
        compared_part = padded_sig[i:i+len(wavelet)]

        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(compared_part)
        # plt.subplot(2, 1, 2)
        # plt.plot(np.real(wavelet))
        # plt.show()

        similarity_results_mass[i] = normalized_cosine_similarity(compared_part,np.real(wavelet))
        # print(compared_part)
        # print(np.real(wavelet))
        # print(similarity_results_mass[i])
        # print('+++++++++++++++++++++++++++++')

    cosine_sim_convolved = similarity_results_mass[1:-wavle_half_pos]
    return cosine_sim_convolved

def normalized_cosine_similarity(vector1, vector2):
    dot_prod = 0
    norm_v1 = 0
    norm_v2 = 0
    for val1, val2 in zip(vector1,vector2):
        if val1 == 0:
            val1 = 0.00001
        if val2 == 0:
            val2 = 0.00001
        #diff_cof = 1
        diff_cof = def_window(val1,val2)
        #diff_cof = rectangular_window(val1, val2, min_bound=0.8, max_bound=1.2)
        #diff_cof = tukey_window(val1,val2,min_bound=0.8,max_bound=1.2)
        dot_prod += (val1*val2)*diff_cof
        norm_v1 +=(val1**2)
        norm_v2+=(val2**2)
        cos_similarity_result = (dot_prod/(np.sqrt(norm_v1)*np.sqrt(norm_v2)))

        # print(dot_prod)
        # print(norm_v1)
        # print(norm_v2)
        # print(cos_similarity_result)
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(vector1)
        # plt.subplot(2, 1, 2)
        # plt.plot(vector2)
        # plt.show()
    return cos_similarity_result

def def_window(val1,val2):
    if val1 == 0 and val2==0:
        return 0
    diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
    diff_cof = diff_cof ** 1
    return diff_cof

def plot(sig, block = True):
    plt.figure()
    plt.plot(sig)
    plt.show(block = block)

def waveletBody(wavelet, amp_norm = None):
    Wavelet_start_count = 0
    Wavelet_end_count = len(wavelet) - 1
    wavelet_real = np.real(wavelet)
    wavelet_max = max(np.abs(wavelet_real))
    treshold = wavelet_max*0.001
    #find_startpos
    for count in range(len(wavelet)):
        if wavelet_real[count] > treshold:
            Wavelet_start_count = count
            break
    #find_endpos
    for count in range(len(wavelet)-1,0,-1):
        if wavelet_real[count] > treshold:
            Wavelet_end_count = count+1
            break

    if amp_norm:
        correction_value = amp_norm/wavelet_max
        wavelet_body = np.array(wavelet)[Wavelet_start_count:Wavelet_end_count] * correction_value
    else:
        wavelet_body = np.array(wavelet)[Wavelet_start_count:Wavelet_end_count]
    return wavelet_body


def fix_length(sig1, sig2):
    if len(sig1)>len(sig2):
        def_l = len(sig1) - len(sig2)
        return np.array(sig1)[0:-def_l], sig2

    if len(sig2)>len(sig1):
        def_l = len(sig2) - len(sig1)
        return sig1, np.array(sig2)[0:-def_l]



def custom_conv_with_metric(sig, wavelet_body):
    print('custom conv')
    sig = np.array(sig)
    wavelet_body = np.array(wavelet_body)
    # Create an empty output mass
    output_mass = [0] * len(sig)
    wavelet_len = len(wavelet_body)
    half_wavelet_offset = ((wavelet_len - 1) / 2).__floor__()
    for sig_point_num in range(len(sig)):
        start_sig_point = max(sig_point_num-half_wavelet_offset,0)
        stop_sig_point = min(sig_point_num+half_wavelet_offset+1,len(sig)-1)
        start_wavelet_point = max(half_wavelet_offset-sig_point_num, 0)
        stop_wavelet_point =  min(half_wavelet_offset + (len(sig)-1 - sig_point_num),wavelet_len-1)

        signal_part_mass = sig[start_sig_point:stop_sig_point+1]
        wavelet_part_mass = wavelet_body[start_wavelet_point:stop_wavelet_point+1]
        if len(signal_part_mass) == len(wavelet_part_mass):
            pass
        else:
            signal_part_mass,wavelet_part_mass = fix_length(signal_part_mass,wavelet_part_mass)
            print(len(signal_part_mass))
            print(len(wavelet_part_mass))
            print(start_sig_point,stop_sig_point,start_wavelet_point,stop_wavelet_point)
            print(wavelet_len)
            print(half_wavelet_offset)
            print ("Signal part and wavelet parts should be equal")

        #signal_part_mass = signal_part_mass - np.mean(signal_part_mass) #make something, need to test on more complicated sig

        #similarity_coef = sf.Cosine_similarity(signal_part_mass, np.real(wavelet_part_mass))
        #similarity_coef = sf.Pearson_Corr(signal_part_mass, np.real(wavelet_part_mass))
        # similarity_coef = sf.normalized_cosine_similarity(signal_part_mass,np.real(wavelet_part_mass))
        similarity_coef = normalized_cosine_similarity(signal_part_mass, np.real(wavelet_part_mass))
        # print(signal_part_mass)
        # print(np.real(wavelet_part_mass))
        # print(similarity_coef)
        # print('---------------------------')
        output_mass[int(sig_point_num)] = similarity_coef


    return output_mass








def cwt_customized(signal, scales):
    n = len(signal)
    custom_cwt_matrix = np.zeros((len(scales), n), dtype='complex_')
    for ind, scale in enumerate(scales):
        wavelet_data = np.conj(morlet_wavelet(n, scale))
        wavelet_body = waveletBody(wavelet_data, amp_norm=0.01)
        print(f'{ind}/{len(scales)}')
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(signal)
        # plt.subplot(2, 1, 2)
        # plt.plot(np.real(wavelet_body))
        # plt.show()
        # custom_cwt_matrix[ind, :] = np.convolve(signal, wavelet_body, mode='same')
        custom_cwt_matrix[ind, :] = custom_conv_with_metric(signal,wavelet_body)
        # print(len(custom_cwt_matrix[ind, :]))
        # input(custom_cwt_matrix[ind, :])

        # custom_cwt_matrix[ind, :] = convolve_cosine_sim_based_mod(signal, wavelet_body)

    # plt.figure()
    # X, Y = np.meshgrid(list(range(len(signal))), scales)
    # plt.pcolormesh(X, Y, np.abs(custom_cwt_matrix), cmap='viridis')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Time')
    # plt.ylabel('Scale')
    # plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    # plt.show()
    return custom_cwt_matrix

def icwt(coefficients, scales):
    c1 = -1
    c2 = np.sqrt(scales)
    c3 = np.real(coefficients)
    return (c3).sum(axis=0)
    return c1*(c3/np.transpose([c2])).sum(axis=0)


fs = 500
sig = sigTotest(fs=fs, part_t=0.1)
t = np.arange(0, len(sig)/fs, 1/fs)
num_scales = 80
scales = np.arange(3, num_scales,1)


# omega0 = 5
# morlet_flambda = (4 * np.pi) / (omega0 + np.sqrt(2 + omega0**2))
# s0 = 2 * (1/fs) / morlet_flambda
# # s0 = 1
# dj = 1/12
#
# sj = s0 * 2 ** (np.arange(0, num_scales + 1) * dj)
# scales = sj

scales = np.arange(1, num_scales)


cwtmatr = cwt_customized(sig, scales)
cwtmatr = np.nan_to_num(cwtmatr, nan=0.0)

rec = icwt(cwtmatr,scales)

plt.figure()
plt.subplot(2,1,1)
plt.plot(sig)
plt.subplot(2,1,2)
plt.plot(rec)
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