from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


class AnalogFilterDesign:
    # ----------In develop!--------------------#
    """
    Creation universal filter
    Example:
        filtered_signal = AnalogFilterDesign(sig,1000).lp(order=5, cutoff=25).zerophaze().bessel().filtration()
        filtered_signal2 = AnalogFilterDesign(sig,1000).bs(order=3, cutoff=[35, 43]).zerophaze().butter().filtration()
    """

    def __init__(self, input_signal, fs):
        self.Signal = input_signal
        self.Fs = fs
        self.Filer = None
        self.Type = None
        self.Order = None
        self.Cutoff = None
        self.Coefs = None
        self.Filer = signal.lfilter
        self.SignalFiltered = None
        self.nyq_rate = self.Fs / 2

    def butter(self):
        b, a = signal.butter(self.Order,
                             self.Cutoff,
                             fs=self.Fs,
                             btype=self.Type,
                             analog=False)
        self.Coefs = {'b': b, 'a': a}
        return self

    def bessel(self):
        b, a = signal.bessel(self.Order,
                             self.Cutoff,
                             fs=self.Fs,
                             btype=self.Type,
                             analog=False)
        self.Coefs = {'b': b, 'a': a}
        return self

    def notch(self, quality_factor=30, cutoff=None):
        self.Cutoff = cutoff
        b, a = signal.iirnotch(self.Cutoff, quality_factor, self.Fs)
        self.Coefs = {'b': b, 'a': a}
        return self

    def lp(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'lowpass'
        self.butter()
        return self

    def hp(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'highpass'
        self.butter()
        return self

    def bp(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'bandpass'
        self.butter()
        return self

    def bs(self, order=None, cutoff=None):
        self.Order = order
        self.Cutoff = cutoff
        self.Type = 'bandstop'
        self.butter()
        return self

    def zerophaze(self):
        self.Filer = signal.filtfilt
        if self.Order:
            self.Order = int(np.ceil(self.Order / 2))
        return self

    def filtration(self, show = False):
        b = self.Coefs['b']
        a = self.Coefs['a']
        self.SignalFiltered = self.Filer(b, a, self.Signal)
        if show:
            self.show_result()
        return self.SignalFiltered

    def freq_resp(self):
        plt.figure()
        plt.clf()
        b = self.Coefs['b']
        a = self.Coefs['a']
        w, h = signal.freqz(b, a, worN=8000)
        plt.plot((w / np.pi) * self.nyq_rate, np.absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.ylim(-0.05, 1.05)
        if type(self.Cutoff) is list:
            for cutoff in self.Cutoff:
                plt.axvline(x=cutoff, color='g', linestyle='-.', linewidth=1)
        else:
            plt.axvline(x=self.Cutoff, color='g', linestyle='-.', linewidth=1)
        plt.grid(True)
        plt.show()

    def show_result(self, block=True):
        def t_vector(sig, fs):
            x = list(range(np.shape(sig)[0]))
            x = np.array(x)
            x = x / fs
            return x

        plt.figure()
        inpit_sig = self.Signal
        inpit_sig_t = t_vector(inpit_sig, self.Fs)
        filtered_sig = self.SignalFiltered
        filtered_sig_t = t_vector(filtered_sig, self.Fs)
        ax1 = plt.subplot(2,1,1)
        plt.plot(inpit_sig_t,inpit_sig,label = 'input', linewidth = 2)
        plt.plot(filtered_sig_t, filtered_sig, label = 'filtered',color = 'r', linewidth = 1)
        plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(filtered_sig_t, filtered_sig, label='filtered', color='r', linewidth=1)
        plt.show(block=block)
        return self
