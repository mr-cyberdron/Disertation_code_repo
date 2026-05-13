import numpy as np
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import Analyse_databases.class_UIselector
import BIOMED_SIGNALS.ECG.ECG_features
from Analyse_databases.modules.WFDB import WfdbParce
from BIOMED_SIGNALS.ECG.class_record import EcgRecord
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pylab
import os
import pandas as pd


def t_vectors(signals, fs) -> list[list[float]]:
    t_vectors_mass = []
    for signal, total_fs in zip(signals, fs):
        sig_units_num = len(signal)
        siglen_s = sig_units_num / total_fs
        t_vector = list(np.linspace(0, siglen_s, sig_units_num))
        t_vectors_mass.append(t_vector)
    return t_vectors_mass

def ui_selector_processing(signals, leads, units, fs_mass, title, filee, metadata,
                           late_atrial_potentials_vector_magnitude_features,
                               late_ventricular_potentials_vector_magnitude_features):
    def detect_type(anot):
        lvp_flag = False
        lap_flag = False
        if not anot:
            return 'no_LAP_no_LVP'
        for event in anot:
            if event['type'] == 'LVP':
                lvp_flag = True
            if event['type'] == 'LAP':
                lap_flag = True
        if lvp_flag == False and lap_flag == True:
            return 'LAP'
        if lvp_flag == True and lap_flag == False:
            return 'LVP'
        if lvp_flag == True and lap_flag == True:
            return 'LAP_and_LVP'


    # -------------------Develop UI--------------------------#
    ui_selector = Analyse_databases.class_UIselector.UIselector()
    ui_selector.signals = signals
    ui_selector.leads = leads
    ui_selector.units = units
    ui_selector.fs_mass = fs_mass
    ui_selector.title = title
    ui_selector.filee = filee
    ui_selector.metadata = metadata

    anotations = ui_selector.select_events()
    typee = detect_type(anotations)
    dataline = {'file': [filee], 'type': [typee], 'metadata_json': [str(metadata)],
                'events': [str(anotations)],
                'LAP_vmagnitude_features': [str(late_atrial_potentials_vector_magnitude_features)],
                'LVP_vmagnitude_features': [str(late_ventricular_potentials_vector_magnitude_features)]}

    df_line = pd.DataFrame(data=dataline)
    old_df = pd.read_csv(logfile_name)
    new_df = pd.concat([old_df, df_line])
    new_df.to_csv(logfile_name, index=False)

    # -------------------------------------------------------#


def plot_ecg_record(signals, leads, units, fs_mass, title, filee, metadata) -> None:
    tvectors = t_vectors(signals, fs_mass)
    n_cols = 3
    if len(signals) < n_cols:
        n_cols = 1
    expected_n_rows = int(np.ceil(len(signals) / n_cols))
    plt.rc('font', size=6)
    fig, axs = plt.subplots(nrows=expected_n_rows, ncols=n_cols, figsize=(14, 6))
    fig.canvas.manager.set_window_title(title)
    counter_rows = 0
    counter_cols = 0

    for signal, lead, unit, tvector in zip(signals, leads, units, tvectors):
        if n_cols == 1 and expected_n_rows == 1:
            plt.plot(tvector, signal, linewidth=0.8)
            plt.legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
            plt.xlabel('[Sec]')
        elif n_cols < 2 or expected_n_rows < 2:
            axs[counter_rows].plot(tvector, signal, linewidth=0.8)
            axs[counter_rows].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
            axs[counter_rows].set_xlabel('[Sec]')
        else:
            if counter_rows > (expected_n_rows - 1):
                counter_cols = counter_cols + 1
                counter_rows = 0
            axs[counter_rows, counter_cols].plot(tvector, signal, linewidth=0.8)  # , color= '#A40483')
            if lead == 'vx':
                axs[counter_rows, counter_cols].set_facecolor('#FFC0B5')
            if lead == 'vy':
                axs[counter_rows, counter_cols].set_facecolor('#FFF4B5')
            if lead == 'vz':
                axs[counter_rows, counter_cols].set_facecolor('#B9FFB5')
            axs[counter_rows, counter_cols].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
            if counter_rows == (expected_n_rows - 1):
                axs[counter_rows, counter_cols].set_xlabel('[Sec]')

        counter_rows = counter_rows + 1

    plt.subplots_adjust(left=0.027, bottom=0.048, right=0.98, top=0.98, wspace=0.082, hspace=0.133)
    plt.show(block=False)
    # part_to_test
    # plt.close(fig)

    if True:  # classif_buttons
        fig2 = plt.figure(figsize=(7, 1))

        def tag_lvp(event):
            save_analyse_result('LVP')

        def tag_lap(event):
            save_analyse_result('LAP')

        def tag_lap_lvp(event):
            save_analyse_result('LVP_and_LAP')

        def tag_skip(event):
            save_analyse_result('none')

        def save_analyse_result(typee):
            dataline = {'file': [filee], 'type': [typee], 'metadata_json': [str(metadata)]}
            df_line = pd.DataFrame(data=dataline)
            old_df = pd.read_csv(logfile_name)
            new_df = pd.concat([old_df, df_line])
            new_df.to_csv(logfile_name, index=False)
            plt.close(fig)
            plt.close(fig2)

        button_LVP = Button(pylab.axes([0.005, 0, 0.25, 1]), 'Late ventricular potentials', color='#698EF1')
        button_LAP = Button(pylab.axes([0.26, 0, 0.25, 1]), 'Late atrial potentials', color='#DA69F1')
        button_LAP_and_LVP = Button(pylab.axes([0.515, 0, 0.25, 1]), 'Late atrial and ventricular potentials',
                                    color='#EDF169')
        button_skip = Button(pylab.axes([0.77, 0, 0.25, 1]), 'Skip', color='#F16969')
        button_LVP.on_clicked(tag_lvp)
        button_LAP.on_clicked(tag_lap)
        button_LAP_and_LVP.on_clicked(tag_lap_lvp)
        button_skip.on_clicked(tag_skip)
        plt.show()


# ------------------------------------Input variables-------------------------------------------#
# ptb-diagnostic-ecg-database-1.0.0 https://physionet.org/content/ptbdb/1.0.0/
file_path_header = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'
# Cerebral Vasoregulation in Elderly with Stroke https://physionet.org/content/cves/1.0.0/
#file_path_header = 'E:/Bases/Cerebral Vasoregulation in Elderly with Stroke/splitting_result/'
# classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0
# https://physionet.org/content/challenge-2016/1.0.0/
#file_path_header = 'E:/Bases/CLASSI~1/CLASSI~1.0/CLASSI~1.0/'
# ----------------------------------------------------------------------------------------------#
files_paths_parts = []
for root, dirs, files in os.walk(file_path_header):
    for file in files:
        if file.endswith('.hea'):
            part_to_add = root.replace(file_path_header, '') + '/' + file.replace('.hea', '')
            files_paths_parts.append(part_to_add)

logfile_name = file_path_header.split('/')[-2] + '_logfile.csv'

if os.path.exists(logfile_name) is False:
    print('not_exist')
    # tmp_df = pd.DataFrame(data={'file': [], 'type': [], 'metadata_json': []})
    tmp_df = pd.DataFrame(data={'file': [], 'type': [], 'metadata_json': [], 'events': []})
    tmp_df.to_csv(logfile_name, index=False)
else:
    print("logfile Exists")

# files_list = list(pd.read_csv(logfile_name)['file'].to_numpy())
files_list = ['patient141/s0307lre'] #'patient045/s0155lre' #patient141/s0307lre #patient045/s0217lre
file_counter = 1
for file in files_paths_parts:
    print(file)
    total_num = str(file_counter) + '/' + str(len(files_paths_parts))
    file_counter += 1
    print(total_num)
    if file in files_list :#and file_counter > 257:  # and file_counter > 181:

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
                                                             'flow_rate', 'o2','co2', 'PCG') #

        # #--------Vector_magnitude_features
        # try:
        #     late_atrial_potentials_vector_magnitude_features, \
        #     late_ventricular_potentials_vector_magnitude_features = \
        #         BIOMED_SIGNALS.ECG.ECG_features.ecg_ortogonal_leads_vector_magnitude_and_features(record_object_data)
        # except:
        #     late_atrial_potentials_vector_magnitude_features = None
        #     late_ventricular_potentials_vector_magnitude_features = None

        last_record_len = list(record_object_data.ged_record_len().values())[-1]
        last_record_fs = record_object_data.Fs[-1]
        print('ECG fs: ' + str(last_record_fs) + ' len:' + str(last_record_len) + ' Sec (' + str(
            round(last_record_len / 60, 4)) + ' min)')
        #record_object_data.saecg_count(remove_strange_cardiocycles=False,avg_method='avg')
        record_object_data.saecg_count(remove_strange_cardiocycles=False, avg_method='custom')
        print(record_object_data.Metadata)
        # signals_p = record_object_data.Signals
        signals_p = record_object_data.SAECG


        leads_p = record_object_data.Leads
        units_p = record_object_data.Units
        fs_mass_p = record_object_data.Fs
        img_tit = file.split('/')[-1] + ' ' + str(file_counter - 1) + ' ' + str(record_object_data.Metadata)
        plot_ecg_record(signals_p, leads_p, units_p, fs_mass_p, img_tit, file, record_object_data.Metadata)


        fig_n = 5 #5
        sig = signals_p[fig_n]
        sig_t = np.array(list(range(len(sig))))/fs_mass_p[fig_n]
        plt.figure(figsize=(6,4))
        plt.subplot(2,1,1)
        plt.plot(sig_t, sig)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        only_lp = AnalogFilterDesign(sig, fs_mass_p[fig_n]).hp(order=5, cutoff=30).zerophaze().butter() \
                .filtration()
        plt.subplot(2, 1, 2)
        plt.plot(sig_t, only_lp)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()


        # ui_selector_processing(signals_p, leads_p, units_p, fs_mass_p, img_tit, file, record_object_data.Metadata,
        #                        late_atrial_potentials_vector_magnitude_features,
        #                        late_ventricular_potentials_vector_magnitude_features)
    else:
        print('skipped')
