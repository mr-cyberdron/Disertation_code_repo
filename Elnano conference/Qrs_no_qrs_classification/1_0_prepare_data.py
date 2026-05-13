import random

import matplotlib.pyplot as plt

import Plottermaan_lib
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb
from FILES_processing_lib import scandir
import numpy as np
from collections import Counter
import json
from Plottermaan_lib import masplot
import tools

qrs_bases_dirs = [
            'E:/Bases/0Prepared_bases/2021_challange_prepared/',
            'E:/Bases/0Prepared_bases/but_pdb_prepared/',
            'E:/Bases/0Prepared_bases/fantasia_prep/',
            'E:/Bases/0Prepared_bases/long-term-af-database_prep/',
            'E:/Bases/0Prepared_bases/mit-bih-arrhythmia-database_prepared/',
            'E:/Bases/0Prepared_bases/mit-bih-long-term-ecg-dat_prep/',
            'E:/Bases/0Prepared_bases/mit-bih-noise-stress-test-database/',
            'E:/Bases/0Prepared_bases/Noised ECG/',
            'E:/Bases/0Prepared_bases/st-petersburg-incart-12-lead-arrhythmia-database_prep/',
            'E:/Bases/0Prepared_bases/t-wave-alternans-challenge-database_prepared/']

noise_bases_dirs = [
    'E:/Bases/0Prepared_bases/Noise/Artif_ecg_sig/',
    'E:/Bases/0Prepared_bases/Noise/BUT_QTB/noise_rec/',
    'E:/Bases/0Prepared_bases/Noise/cu-ventricular-tachyarrhythmia-database/',
    'E:/Bases/0Prepared_bases/Noise/mit-bih-arrhythmia-database/',
    'E:/Bases/0Prepared_bases/Noise/mit-bih-noise-stress-test-database/'
]

out_dir = 'E:/Bases/1QRS_noise_base/'


allowed_beat_types = ["N","V","A","F","a","E","R",
                      "/","j","L","f","!","S","r","e","n"]


def prepare_QRS(qrs_bases_dirs, allowed_beat_types):
    for base in qrs_bases_dirs:
        base_name = base.split('/')[-2]
        fnames = tools.get_wfdb_names(base)
        file_counter = 0
        prepared_qrs_mass = np.array([])
        for fname in fnames:
            file_counter +=1
            print(f'prep_QRS_{base_name}_{file_counter}/{len(fnames)}')
            signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(base+fname)

            qrs_samples = tools.prep_qrs_data(signals, fs, annotations_pos, qrs_annotations, allowed_beat_types)

            # signals = tools.filter_signals(signals,fs, type='hp') #baseline removal
            # signals = tools.normalize_ecg_signals(signals,fs,annotations_pos)
            # qrs_samples = tools.select_qrs_samples(signals,fs,annotations_pos,qrs_annotations, allowed_beat_types)
            if len(qrs_samples)>0:
                if not np.any(prepared_qrs_mass):
                    prepared_qrs_mass = qrs_samples
                else:
                    prepared_qrs_mass = np.concatenate((prepared_qrs_mass,qrs_samples),axis=0)

        print('saving')
        print('qrs')
        print(f'{base_name}:{len(prepared_qrs_mass)}')
        np.savez(out_dir + f'{base_name}_QRS_X.npz', prepared_qrs_mass)
        qrs_labels = np.ones(len(prepared_qrs_mass))
        np.savez(out_dir + f'{base_name}_QRS_y.npz', qrs_labels)


def prepare_noise(Noise_bases_dirs):
    for base in Noise_bases_dirs:
        base_name = base.split('/')[-2]
        fnames = tools.get_wfdb_names(base)
        file_counter = 0
        prepared_noise_mass = np.array([])
        for fname in fnames:
            file_counter += 1
            print(f'Noise_{base_name}_{file_counter}/{len(fnames)}')
            signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(base + fname)
            fake_annotations_pos, fake_annotations = tools.generate_fake_qrs(signals,fs)
            if len(fake_annotations_pos)>0:
                signals = tools.filter_signals(signals, fs, type='hp')  # baseline removal
                signals = tools.normalize_ecg_signals(signals, fs, fake_annotations_pos)
                noise_samples = tools.select_qrs_samples(signals, fs, fake_annotations_pos, fake_annotations,
                                                       ['N'])

                if len(noise_samples) > 0:
                    if not np.any(prepared_noise_mass):
                        prepared_noise_mass = noise_samples
                    else:
                        prepared_noise_mass = np.concatenate((prepared_noise_mass,noise_samples), axis=0)
        print('saving')
        print('noise')
        print(f'{base_name}:{len(prepared_noise_mass)}')
        np.savez(out_dir + f'{base_name}_Noise_X.npz', prepared_noise_mass)
        noise_labels = np.zeros(len(prepared_noise_mass))
        np.savez(out_dir + f'{base_name}_Noise_y.npz', noise_labels)





# prepare_QRS(qrs_bases_dirs, allowed_beat_types)
prepare_noise(noise_bases_dirs)







