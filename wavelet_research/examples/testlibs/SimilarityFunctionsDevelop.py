import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import math
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

def Pearson_Corr(sig1, sig2):
    if len(sig1)>1 and len(sig2)>1:
        return np.corrcoef(sig1,sig2)[0][1]
    else:
        return 0

def Cosine_similarity(sig1, sig2):
    if len(sig1)>1 and len(sig2)>1:
        from scipy.spatial.distance import cosine
        return np.abs(1 - cosine(sig1, sig2))
    else:
        return 0

def Euclidian_distance(sig1, sig2):
    if len(sig1)>1 and len(sig2)>1:
        return np.linalg.norm(np.array(sig1) - np.array(sig2))
    else:
        return 0

def def_window(val1,val2):
    diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
    diff_cof = diff_cof ** 1
    return diff_cof

def rectangular_window(val1,val2,min_bound = 0.8, max_bound = 1.2):
    diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
    if diff_cof>min_bound and diff_cof<max_bound:
        return 1
    else:
        return 0

def tukey_window(val1,val2, min_bound = 0.8, max_bound = 1.2,alpha = 0.5):
    n_counts = 100
    shift_vector = np.arange(min_bound,max_bound, (max_bound-min_bound)/n_counts)
    window = signal.windows.tukey(n_counts, alpha=alpha)
    diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
    window_correct_coef = window[np.where(shift_vector>diff_cof)][0]
    diff_cof = window_correct_coef
    # plt.plot(shift_vector,window)
    # plt.title("Tukey window")
    # plt.ylabel("Amplitude")
    # plt.xlabel("Sample")
    # plt.ylim([0, 1.1])
    # plt.show()
    return diff_cof


def normalized_cosine_similarity(vector1, vector2):
    dot_prod = 0
    norm_v1 = 0
    norm_v2 = 0
    for val1, val2 in zip(vector1,vector2):
        #diff_cof = 1
        diff_cof = def_window(val1,val2)
        #diff_cof = rectangular_window(val1, val2, min_bound=0.8, max_bound=1.2)
        #diff_cof = tukey_window(val1,val2,min_bound=0.8,max_bound=1.2)
        dot_prod += (val1*val2)*diff_cof**1
        norm_v1 +=(val1**2)
        norm_v2+=(val2**2)
    return dot_prod/(np.sqrt(norm_v1)*np.sqrt(norm_v2))


