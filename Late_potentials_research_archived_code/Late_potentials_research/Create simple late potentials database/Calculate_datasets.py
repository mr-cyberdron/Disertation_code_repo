import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from Frequency_tools.Filtering import AnalogFilters
import BIOMED_SIGNALS.ECG.ECG_features
import Withdrawal_tools
from Analyse_databases.modules.WFDB import WfdbParce
from BIOMED_SIGNALS.ECG.class_record import EcgRecord
import numpy as np
from scipy import signal

def create_files_paths(file_path_header, files_names_to_search):
    files_paths_parts = []
    for root, dirs, files in os.walk(file_path_header):
        for file in files:
            if file.endswith('.hea'):
                part_to_add = root.replace(file_path_header, '') + '/' + file.replace('.hea', '')
                if str(part_to_add) in files_names_to_search:
                    files_paths_parts.append(root + '/' + file)
    return files_paths_parts

def saecg_filter(saecg):
    if not list(saecg):
        return None
    else:
        filtered_saecg = []
        saecg_lens_mas = []
        for saecg_item in saecg:
            if saecg_item == [0, 0, 0]:
                pass
            else:
                filtered_saecg.append(saecg_item)
                saecg_lens_mas.append(len(saecg_item))
        from statistics import mode
        if not list(filtered_saecg):
            return None
        mode_saecg_len = mode(saecg_lens_mas)
        filtered2_saecg = []
        for saecg_filt_item in filtered_saecg:
            if len(saecg_filt_item) != mode_saecg_len:
                pass
            else:
                filtered2_saecg.append(saecg_filt_item)
        if not list(filtered2_saecg):
            return None
        return filtered2_saecg

def calc_VM(saecg_mas, fs):
    if saecg_mas is not None:
        saecg_power_sum = []
        for  saecg in saecg_mas:
            saecg = np.array(saecg) - np.mean(saecg)
            saecg = AnalogFilters.AnalogFilterDesign(saecg, fs).hp(order=5, cutoff=40).zerophaze().butter() \
                .filtration()
            saecg_powered = np.array(np.power(saecg, 2))
            if not list(saecg_power_sum):
                saecg_power_sum = saecg_powered
            else:
                saecg_power_sum = saecg_power_sum + saecg_powered

        return np.sqrt(saecg_power_sum)
    else:
        return None

def time_frequency_energy(vm, fs, from_count, to_count, from_fr, to_fr):
    if list(vm) and from_count and to_count:
        f, t, Sxx = signal.spectrogram(vm, fs, nperseg= 64, noverlap=30)
        freq_part = Sxx[np.where((f>=from_fr)& (f<=to_fr))]
        try:
            t_start = np.where(t<=(from_count/fs))[0][-1]
            t_stop = np.where(t>=(to_count/fs))[0]
        except:
            return None
        if not list(t_stop):
            t_stop = len(t)-1
        else:
            t_stop = t_stop[0]
        t_and_freq_part = freq_part[:, t_start:t_stop]
        t_and_freq_part_rows = int(np.shape(t_and_freq_part)[0])
        t_and_freq_part_cols = int(np.shape(t_and_freq_part)[1])
        k = t_and_freq_part_rows * t_and_freq_part_cols
        t_f_part_power = np.power(t_and_freq_part, 2)
        E = np.sum(t_f_part_power, axis=0)
        E = np.sum(E)
        E = E/k
        return E
    else:
        return None



def calc_lvp_features(vm, fs, units):
    if vm is not None:
        vm = BIOMED_SIGNALS.ECG.ECG_features.normalize_lead_signal(vm, units)
        vm_features = BIOMED_SIGNALS.ECG.ECG_features.saecg_vector_magnitude_detect_features(vm, fs)
        #F1
        if vm_features['las_40_point'] and vm_features['r_peak_pos']:
            R_J_lag = vm_features['las_40_point'] - vm_features['r_peak_pos']
            if R_J_lag < 0:
                R_J_lag = None
            else:
                R_J_lag = R_J_lag/fs
        else:
            R_J_lag = None
        #F2
        rms_40 = vm_features['rms_40_uV']
        #F3
        las_40 = vm_features['las_40_ms']
        if las_40 == -1:
            las_40 = None
        # calc Time-frequency energy
        print(vm_features)
        if vm_features['qrs_offset']:
            E_end = time_frequency_energy(vm,fs,vm_features['qrs_offset'],(vm_features['qrs_offset']+ (0.08*fs)),55, 300)
        else:
            E_end = None
        if vm_features['qrs_onset'] and vm_features['qrs_offset']:
            E_QRS = time_frequency_energy(vm,fs,vm_features['qrs_onset'],(vm_features['qrs_offset']),55, 300)
        else:
            E_QRS = None
        if vm_features['las_40_point']:
            E_VLP = time_frequency_energy(vm,fs,(vm_features['las_40_point'] - (0.055*fs)),(vm_features['las_40_point'] + (0.025*fs)),55, 300)
        else:
            E_VLP = None
        #F4
        if E_VLP and E_QRS:
            F4 = E_VLP/E_QRS
        else:
            F4 = None
        if E_end and E_QRS:
            F5 = E_end/E_QRS
        else:
            F5 = None

        return {'R_J_lag':R_J_lag, 'rms_40': rms_40, 'las_40': las_40, 'E_VLP/E_QRS': F4, 'E_end/E_QRS': F5}
    else:
        return {'R_J_lag':None, 'rms_40': None, 'las_40': None, 'E_VLP/E_QRS': None, 'E_end/E_QRS': None}


def calc_lap_features(vm, fs, units):
    if vm is not None:
        vm = BIOMED_SIGNALS.ECG.ECG_features.normalize_lead_signal(vm, units)
        vm_features = BIOMED_SIGNALS.ECG.ECG_features.psaecg_vector_magnitude_detect_features(vm, fs)
        #F1
        if vm_features['p_offset'] and vm_features['p_peak_pos']:
            p_offset_lag = vm_features['p_offset'] - vm_features['p_peak_pos']
            if p_offset_lag < 0:
                p_offset_lag = None
            else:
                p_offset_lag = p_offset_lag/fs
        else:
            p_offset_lag = None
        #F2
        rms_20 = vm_features['rms_20_uV']
        #F3
        E_end = time_frequency_energy(vm, fs, vm_features['p_offset'], (vm_features['p_offset'] + (0.08 * fs)), 55, 300)
        E_P = time_frequency_energy(vm, fs, vm_features['p_onset'], (vm_features['p_offset']), 55, 300)
        if E_end and E_P:
            F3 = E_end/E_P
        else:
            F3 = None
        #F4
        p_dur = vm_features['p_dur_ms']

        return {'p_offset_lag':p_offset_lag, 'rms_20': rms_20, 'E_end/E_P': F3, 'p_dur':p_dur}
    else:
        return {'p_offset_lag': None, 'rms_20': None, 'E_end/E_P': None, 'p_dur': None}






file_path_header = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'
late_potentials_metadata = pd.read_csv('ptb-diagnostic-ecg-database-1.0.0_logfile.csv')
files_names = list(late_potentials_metadata['file'].to_numpy())
late_potentials_characteristics = list(late_potentials_metadata['metadata_json'].to_numpy())

LVP_data = []
LAP_data = []
LVP_and_LAP_data = []
No_LVP_no_LAP = []

for file_name, characteristic in zip(files_names, late_potentials_characteristics):
    characteristic_data = dict(json.loads(characteristic))

    rms_20_stat = characteristic_data['lap_simps']['rms_20'][-1]
    p_dur_stat = characteristic_data['lap_simps']['p_dur'][-1]

    las_40_stat = characteristic_data['lvp_simps']['las_40'][-1]
    rms_40_stat = characteristic_data['lvp_simps']['rms_40'][-1]
    qrs_dur_stat = characteristic_data['lvp_simps']['qrs_dur'][-1]

    lap_state_counter = 0
    if rms_20_stat:
        lap_state_counter +=1
    if p_dur_stat:
        lap_state_counter +=1

    lvp_state_counter = 0
    if las_40_stat:
        lvp_state_counter+=1
    if rms_40_stat:
        lvp_state_counter +=1
    if qrs_dur_stat:
        lvp_state_counter+=1

    lvp_treshold = 2
    lap_treshold = 2

    if lap_state_counter > (lap_treshold-1) and lvp_state_counter < lvp_treshold:
        LAP_data.append(file_name)
    if lap_state_counter <lap_treshold and lvp_state_counter > (lvp_treshold-1):
        LVP_data.append(file_name)
    if lap_state_counter > (lap_treshold-1) and lvp_state_counter > (lvp_treshold-1):
        LVP_and_LAP_data.append(file_name)
    if lap_state_counter <lap_treshold and lvp_state_counter < lvp_treshold:
        No_LVP_no_LAP.append(file_name)

print('Lvp data ' + str(len(LVP_data)))
print('Lap data ' + str(len(LAP_data)))
print('Lap and LVP data ' + str(len(LVP_and_LAP_data)))
print('No Lap and LVP data ' + str(len(No_LVP_no_LAP)))

def calc_features(file_paths, mode = 'LAP', avg = 'custom' , tag = None):
    try:
        files_processed = Withdrawal_tools.txt_log_read('./LOG.txt')
    except:
        files_processed = []
    counter = 1
    for file in file_paths:
        try:
            print(str(counter)+'/'+str(len(file_paths)))
            counter = counter+1
            if file not in files_processed:
                wfdb_file_object = WfdbParce(file).read()
                record_object_data = EcgRecord(Fs=wfdb_file_object.Fs,
                                               Signals=wfdb_file_object.Signals,
                                               Leads=wfdb_file_object.Leads,
                                               Units=wfdb_file_object.Units,
                                               Metadata=wfdb_file_object.Metadata)
                record_object_data = record_object_data.remove_leads('marker', 'abp', 'mcar', 'mcal', 'radi', 'thermst',
                                                                     'flow_rate', 'o2', 'co2', 'PCG',
                                                                     #'avr', 'avl', 'avf',  'v2', 'v3', 'v5',
                                                                     'vx', 'vy', 'vz',
                                                                    #'i','ii','iii','v1', 'v4', 'v6'
                                                                     )
                if mode == 'LVP':
                    record_object_data.saecg_count(avg_method=avg, only_lp=False)
                    saecg = record_object_data.SAECG
                    saecg_filtered  = saecg_filter(saecg)
                    saecg_vm = calc_VM(saecg_filtered, record_object_data.Fs[0])
                    lvp_features = calc_lvp_features(saecg_vm, record_object_data.Fs[0], record_object_data.Units[0])
                    lvp_features['file'] = file
                    lvp_features['class'] = tag

                    dataset_row = {}
                    for keys, values in zip(list(lvp_features.keys()), list(lvp_features.values())):
                        dataset_row[keys] = [values]
                    Features_dataset_row = pd.DataFrame(data=dataset_row)

                if mode == 'LAP':
                    record_object_data.psaecg_count(avg_method=avg, only_lp=False)
                    psaecg = record_object_data.pSAECG
                    psaecg_filtered = saecg_filter(psaecg)
                    psaecg_vm = calc_VM(psaecg_filtered, record_object_data.Fs[0])
                    lap_features = calc_lap_features(psaecg_vm, record_object_data.Fs[0],record_object_data.Units[0])
                    lap_features['file'] = file
                    lap_features['class'] = tag
                    dataset_row = {}
                    for keys, values in zip(list(lap_features.keys()), list(lap_features.values())):
                        dataset_row[keys] = [values]
                    Features_dataset_row = pd.DataFrame(data=dataset_row)
                print(dataset_row)
                try:
                    old_dataset = pd.read_csv('./features_' + str(mode)+ '_avg_'+ str(avg)+ '_'+str(tag) + '_'+'_.csv')
                    new_dataset = pd.concat([old_dataset,Features_dataset_row], axis=0)
                    new_dataset.to_csv('./features_' + str(mode)+ '_avg_'+ str(avg)+ '_'+str(tag) + '_'+'_.csv', index=False)
                except:
                    Features_dataset_row.to_csv('./features_' + str(mode)+ '_avg_'+ str(avg)+ '_'+str(tag) + '_'+'_.csv', index=False)

                Withdrawal_tools.txt_log(file,'./LOG.txt')
        except:
            pass



def create_dataset(normal_file_names, patology_file_names, nornal_prefix, patology_prefix, mode = 'LAP', avg = 'custom'):
    normal_paths = create_files_paths(file_path_header, normal_file_names)
    patology_paths =  create_files_paths(file_path_header, patology_file_names)
    limit = None
    if limit:
        normal_paths = normal_paths[0:limit]
        patology_paths = patology_paths[0:limit]

    features_dataset_normal = calc_features(normal_paths, mode=mode, avg= avg, tag = 'Normal')
    #features_dataset_normal = calc_features(patology_paths, mode=mode, avg=avg, tag='Patology')



#LVP dataset
#create_dataset(LAP_data+No_LVP_no_LAP,LVP_data+LVP_and_LAP_data, 'NO_LVP_DATA', 'LVP_DATA', mode='LVP', avg = 'custom')
#create_dataset(LAP_data+No_LVP_no_LAP,LVP_data+LVP_and_LAP_data, 'NO_LVP_DATA', 'LVP_DATA', mode='LVP', avg = 'avg')
#
# #LAP dataset
#create_dataset(LVP_data+No_LVP_no_LAP,LAP_data+LVP_and_LAP_data, 'NO_LAP_DATA', 'LAP_DATA', mode='LAP', avg = 'custom')
create_dataset(LVP_data+No_LVP_no_LAP,LAP_data+LVP_and_LAP_data, 'NO_LAP_DATA', 'LAP_DATA', mode='LAP', avg = 'avg')

