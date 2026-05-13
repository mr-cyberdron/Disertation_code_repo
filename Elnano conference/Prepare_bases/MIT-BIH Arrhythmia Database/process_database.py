import FILES_processing_lib
import numpy as np
from process_peculiarities import process_peculiarities
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb


db_source = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/mit-bih-arrhythmia-database-1.0.0/'
storing_path = 'E:/Bases/0Prepared_bases/mit-bih-arrhythmia-database_prepared/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'atr')

    files_with_pecularities = list(process_peculiarities.keys())
    if file in files_with_pecularities:
        if process_peculiarities[file][0] == '1_lead_only':
            signals = [signals[0]]
            re_format = [re_format[0]]
            units = [units[0]]
            adc_gain = [adc_gain[0]]
            baseline = [baseline[0]]
            lead_names = [lead_names[0]]


    write_annotated_wfdb(file.replace('.hea',''),np.array(signals).T,fs,re_format,units,adc_gain,baseline,coments,lead_names,
                        annotations_pos,qrs_annotations,ecg_events,p=storing_path)



