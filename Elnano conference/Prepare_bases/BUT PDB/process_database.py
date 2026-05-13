import FILES_processing_lib
import numpy as np
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb


db_source = 'E:/Bases/brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0 (1)/'
storing_path = 'E:/Bases/0Prepared_bases/but_pdb_prepared/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'qrs')


    write_annotated_wfdb(file.replace('.hea',''),np.array(signals).T,fs,re_format,units,adc_gain,baseline,coments,lead_names,
                        annotations_pos,qrs_annotations,ecg_events,p=storing_path)



