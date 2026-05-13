from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np


def xyplot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)

def fftplot(x,y, fs):
    N = len(x)
    T = 1/fs
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    plt.figure()
    plt.stem(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()



Fs = 10000

RPPt = np.arange(0.201,(0.201+0.028),(1/Fs))
RPP = 0.005*np.sin(2*np.pi*78*RPPt)+0.012*np.sin(2*np.pi*116*RPPt+(np.pi/4))+ 0.002*np.sin(2*np.pi*102*RPPt+(np.pi/2))


xyplot(RPPt,RPP)
fftplot(RPPt,RPP, Fs)
plt.show()

