import FILES_processing_lib
import numpy as np
from process_peculiarities import peculiarities
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb

def clear_anot(annotations_pos, qrs_annotations, ecg_events, clearfrom_sample,clear_to_sample):
    new_pos = []
    new_qrs_anot = []
    new_events_anot = []
    for pos, qrs,event in zip(annotations_pos,qrs_annotations,ecg_events):
        if pos>=clearfrom_sample and pos<=clear_to_sample:
            pass
        else:
            new_pos.append(pos)
            new_qrs_anot.append(qrs)
            new_events_anot.append(event)
    return np.array(new_pos),new_qrs_anot,new_events_anot

db_source = 'E:/Bases/mit-bih-noise-stress-test-database-1.0.0/mit-bih-noise-stress-test-database-1.0.0/'
storing_path = 'E:/Bases/0Prepared_bases/mit-bih-noise-stress-test-database/'
noise_samps = [[108540,149673],[194744,237630],[280973,324253],[367567,410667],[453471,497634],[539957,583416],[626480,649932]]
Noise_sigs = ['bw.hea','em.hea','ma.hea']
files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    if file not in Noise_sigs:
        total_file_num = total_file_num+1
        print(f'{total_file_num}/{len(files)}')
        print(file)
        signals, re_format, fs, units, adc_gain, baseline, \
        coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'atr')
        if file in list(peculiarities.keys()):
            if peculiarities[file] == 'Noise':
                for noise_pos in noise_samps:
                    annotations_pos, qrs_annotations, ecg_events = clear_anot(annotations_pos, qrs_annotations, ecg_events,noise_pos[0],noise_pos[1])

        write_annotated_wfdb(file.replace('.hea', ''), np.array(signals).T, fs, re_format, units, adc_gain, baseline,
                             coments, lead_names,
                             annotations_pos, qrs_annotations, ecg_events, p=storing_path)