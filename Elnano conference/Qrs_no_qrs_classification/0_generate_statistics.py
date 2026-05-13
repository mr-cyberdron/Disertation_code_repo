import Plottermaan_lib
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb
from FILES_processing_lib import scandir
import numpy as np
from collections import Counter
import json
from Plottermaan_lib import masplot

def plot_beat(signals,fs, beat_pos, beat_ype):
    boundaries_sec = 1
    boundaries_samples = boundaries_sec*fs
    sample_from = beat_pos-boundaries_samples
    if sample_from <0:
        sample_from = 0
    sample_to = beat_pos+boundaries_samples

    signals_fragm = signals[:,sample_from:sample_to]

    Plottermaan_lib.masplot2(signals_fragm, markers={'s':boundaries_samples-1}, fs=fs)
    input('s')


def save_json(dict, p = './p1.json'):
    with open(p, 'w') as json_file:
        json.dump(dict, json_file)

qrs_bases_dirs = ['E:/Bases/0Prepared_bases/2021_challange_prepared/',
            'E:/Bases/0Prepared_bases/but_pdb_prepared/',
            'E:/Bases/0Prepared_bases/fantasia_prep/',
            'E:/Bases/0Prepared_bases/long-term-af-database_prep/',
            'E:/Bases/0Prepared_bases/mit-bih-arrhythmia-database_prepared/',
            'E:/Bases/0Prepared_bases/mit-bih-long-term-ecg-dat_prep/',
            'E:/Bases/0Prepared_bases/mit-bih-noise-stress-test-database/',
            'E:/Bases/0Prepared_bases/Noised ECG/',
            'E:/Bases/0Prepared_bases/st-petersburg-incart-12-lead-arrhythmia-database_prep/',
            'E:/Bases/0Prepared_bases/t-wave-alternans-challenge-database_prepared/']


qrs_annotations_mass = []
events_annotations_mass =[]
for base_dir in qrs_bases_dirs:
    files_in_dr = scandir(base_dir,ext='.hea')
    file_counter = 0
    for file in files_in_dr:
        if True:
        # try:
            file_counter = file_counter+1
            print(f'{file_counter}/{len(files_in_dr)}')
            signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(base_dir+file)
            for ann, pos in zip(qrs_annotations, annotations_pos):
                qrs_annotations_mass.append(ann)
                if ann == "|":
                    plot_beat(signals,fs,pos,ann)
            for ev, pos2 in zip(ecg_events, annotations_pos):
                events_annotations_mass.append(ev)
        # except:
        #     print('exeption')
qrs_counts = dict(Counter(qrs_annotations_mass))
events_counts = dict(Counter(events_annotations_mass))

print(qrs_counts)
print(events_counts)

save_json(qrs_counts,'./qrs_stat.json')
save_json(events_counts,'./events_stat.json')

