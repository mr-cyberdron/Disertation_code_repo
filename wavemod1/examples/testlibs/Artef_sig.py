import numpy as np
import matplotlib.pyplot as plt

def doublesin(sin1_hz,sin2_hz,fs,dur, plotflag = False):
    t = np.arange(0, dur, 1/fs)  # Time vector

    signal1 = np.sin(2 * np.pi * sin1_hz * t)
    signal2 = np.sin(2 * np.pi * sin2_hz * t)
    # Combine the two sinusoids
    combined_signal = signal1 + signal2
    if plotflag:
        # Plotting the signal
        plt.figure(figsize=(10, 6))
        plt.plot(t, combined_signal)
        plt.title('Combined Sinusoidal Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    return combined_signal, t

def sigTotest(part_t = 0.2, fs = 1000):
    t = np.arange(0, part_t, 1 / fs)
    signal_p1_1 = np.sin(2 * np.pi * 30 * t)
    signal_p1_2 = np.sin(2 * np.pi * 100 * t)
    signal_p1 = signal_p1_1+signal_p1_2
    signal_p2 = [0]*len(t)
    signal_p3 = np.sin(2 * np.pi * 30 * t)
    signal_p4 = [0] * len(t)
    signal_p5 = np.sin(2 * np.pi * 30 * t)
    for i in range(int(np.round(len(signal_p5)*0.25))):
        signal_p5[np.random.choice(np.arange(0,len(signal_p5)-1))] = 0

    start_freq = 1  # Начальная частота (в герцах)
    end_freq = 10  # Конечная частота (в герцах)
    t = np.arange(0, part_t*2, 1 / fs)
    frequency = np.linspace(start_freq, end_freq, len(t))
    signal_p6= [0] * len(t)
    signal_p7 = np.sin(2 * np.pi * frequency * t)
    signal_p8 = [0] * len(t)
    resulted_sig = np.concatenate((signal_p1*3, signal_p2, signal_p3,signal_p4,signal_p5*3,signal_p6,signal_p7,signal_p8,signal_p3*0.002))
    return resulted_sig


def sin(f0,amp = 1,t_sec = 1, fs = 1000):
    t = np.arange(0, t_sec, 1 / fs)
    return np.sin(2 * np.pi * f0 * t)*amp


def break_sin(f0,amp = 1,t_sec = 1, fs = 1000, break_percent = 10):
    t = np.arange(0, t_sec, 1 / fs)
    signal_p5 = np.sin(2 * np.pi * f0 * t) * amp
    break_counts_num = int(np.round(len(signal_p5)*0.01*break_percent))
    for i in range(break_counts_num):
        signal_p5[np.random.choice(np.arange(0, len(signal_p5) - 1))] = 0
    return signal_p5

def signal_zerofill_left(input_sig,fs,t_sec):
    zerosig = [0] * int(np.round(t_sec*fs))
    return np.concatenate((zerosig,input_sig))

def signal_zerofill_right(input_sig,fs,t_sec):
    zerosig = [0] * int(np.round(t_sec * fs))
    return np.concatenate((input_sig,zerosig))








