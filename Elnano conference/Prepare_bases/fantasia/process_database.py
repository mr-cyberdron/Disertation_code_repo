import FILES_processing_lib
import numpy as np
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb

#-------------------Add noise detection need to add--------------#

db_source = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/fantasia-database-1.0.0/'
storing_path = 'E:/Bases/0Prepared_bases/fantasia_prep/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'ecg')

    ecg_ind = np.where(np.array(lead_names) == 'ECG')[0][0]
    signals = [signals[ecg_ind]]
    re_format = [re_format[ecg_ind]]
    units = [units[ecg_ind]]
    adc_gain = [adc_gain[ecg_ind]]
    baseline = [baseline[ecg_ind]]
    lead_names = [lead_names[ecg_ind]]

    new_events = []
    for event in ecg_events:
        if event == '':
            new_events.append('')
        else:
            new_events.append('')


    write_annotated_wfdb(file.replace('.hea',''),np.array(signals).T,fs,re_format,units,adc_gain,baseline,coments,lead_names,
                        annotations_pos,qrs_annotations,new_events,p=storing_path)



