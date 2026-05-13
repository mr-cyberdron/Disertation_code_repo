import json
import os
import pandas as pd
import numpy as np

import Analyse_databases.class_UIselector
import BIOMED_SIGNALS.ECG.ECG_features
from Analyse_databases.modules.WFDB import WfdbParce
from BIOMED_SIGNALS.ECG.class_record import EcgRecord
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pylab
import os
import pandas as pd
from multiprocessing import Pool

# ------------------------------------Input variables-------------------------------------------#
# ptb-diagnostic-ecg-database-1.0.0 https://physionet.org/content/ptbdb/1.0.0/
file_path_header = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'

files_paths_parts = []
for root, dirs, files in os.walk(file_path_header):
    for file in files:
        if file.endswith('.hea'):
            part_to_add = root.replace(file_path_header, '') + '/' + file.replace('.hea', '')
            files_paths_parts.append(part_to_add)

logfile_name = file_path_header.split('/')[-2] + '_logfile.csv'

if os.path.exists(logfile_name) is False:
    print('not_exist')
    tmp_df = pd.DataFrame(data={'file': [], 'type': [], 'metadata_json': [], 'events': []})
    tmp_df.to_csv(logfile_name, index=False)
else:
    print("logfile Exists")

files_list = list(pd.read_csv(logfile_name)['file'].to_numpy())

file_counter = 1
for file in files_paths_parts:
    print(file)
    total_num = str(file_counter) + '/' + str(len(files_paths_parts))
    file_counter += 1
    print(total_num)
    if file not in files_list :#and file_counter > 257:  # and file_counter > 181:
        try:
            # direct_file_name = file.split('/')[-1]
            # if direct_file_name == "s0090lre" +'\n':
            full_record_path = file_path_header + file.replace('\n', '')
            wfdb_file_object = WfdbParce(full_record_path).read()
            record_object_data = EcgRecord(Fs=wfdb_file_object.Fs,
                                           Signals=wfdb_file_object.Signals,
                                           Leads=wfdb_file_object.Leads,
                                           Units=wfdb_file_object.Units,
                                           Metadata=wfdb_file_object.Metadata)
            record_object_data = record_object_data.remove_leads('marker', 'abp', 'mcar', 'mcal', 'radi', 'thermst',
                                                                 'flow_rate', 'o2', 'co2', 'PCG')

            late_atrial_potentials_vector_magnitude_features, \
            late_ventricular_potentials_vector_magnitude_features = \
                BIOMED_SIGNALS.ECG.ECG_features.ecg_ortogonal_leads_vector_magnitude_and_features(record_object_data, avg = 'custom') #avg = 'avg'

            last_record_len = list(record_object_data.ged_record_len().values())[-1]
            last_record_fs = record_object_data.Fs[-1]
            print('ECG fs: ' + str(last_record_fs) + ' len:' + str(last_record_len) + ' Sec (' + str(
                round(last_record_len / 60, 4)) + ' min)')

            vector_magnitude_features_dump_line = json.dumps({'lap_simps': late_atrial_potentials_vector_magnitude_features,
                                                              'lvp_simps': late_ventricular_potentials_vector_magnitude_features})

            df_data = pd.DataFrame(data={'file': [str(file)], 'type': [None], 'metadata_json': [str(vector_magnitude_features_dump_line)]})

            old_df = pd.read_csv(logfile_name)
            new_df = pd.concat([old_df, df_data],axis=0)
            new_df.to_csv(logfile_name, index=False)
        except:
            pass
