from Analyse_databases.modules import PDF
from Analyse_databases.modules import FILES_processing_lib
from Analyse_databases.modules import DOCX
import json
import numpy as np
from Analyse_databases.modules.WFDB import WfdbParce
from BIOMED_SIGNALS.ECG.class_record import EcgRecord
import matplotlib.pyplot as plt
import pandas as pd
import os

# ------------------------------------Input variables-------------------------------------------#
logfile_path_0 = './old_logs/logfile.csv'
file_path_header_0 = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'
classes_dict_0 = {"LVP": 'Late Ventricular Potentials',
                  'LAP': 'Late Atrial Potentials',
                  'LVP_and_LAP': 'Late Ventricular and Atrial Potentials',
                  'none': 'No_Late_Potentials'}
# ----------------------------------------------------------------------------------------------#

def t_vectors(signals, fs) -> list[list[float]]:
    t_vectors_mass = []
    for signal, total_fs in zip(signals, fs):
        sig_units_num = len(signal)
        siglen_s = sig_units_num / total_fs
        t_vector = list(np.linspace(0, siglen_s, sig_units_num))
        t_vectors_mass.append(t_vector)
    return t_vectors_mass


def save_ecg_record(signals, leads, units, fs_mass, title):
    tvectors = t_vectors(signals, fs_mass)
    n_cols = 3
    expected_n_rows = int(np.ceil(len(signals) / n_cols))
    plt.rc('font', size=6)
    fig, axs = plt.subplots(nrows=expected_n_rows, ncols=n_cols, figsize=(14, 9.5))
    fig.canvas.manager.set_window_title(title)
    counter_rows = 0
    counter_cols = 0
    for signal, lead, unit, tvector in zip(signals, leads, units, tvectors):
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

    plt.subplots_adjust(left=0.027, bottom=0.06, right=0.98, top=0.98, wspace=0.082, hspace=0.185)
    plt.savefig(title, dpi=500)
    plt.close('all')
    return title


def create_saecg_img_from_data(file_path):
    wfdb_file_object = WfdbParce(file_path).read()
    record_object_data = EcgRecord(Fs=wfdb_file_object.Fs,
                                   Signals=wfdb_file_object.Signals,
                                   Leads=wfdb_file_object.Leads,
                                   Units=wfdb_file_object.Units,
                                   Metadata=wfdb_file_object.Metadata)
    record_object_data.saecg_count()
    print('Save graph for' + file_path)
    # signals_p = record_object_data.Signals
    signals_p = record_object_data.SAECG
    leads_p = record_object_data.Leads
    units_p = record_object_data.Units
    fs_mass_p = record_object_data.Fs
    FILES_processing_lib.create_floder('./results/tmp/')
    tosave_path = './results/tmp/' + file_path.split('/')[-1] + '.jpg'
    saved_path = save_ecg_record(signals_p, leads_p, units_p, fs_mass_p, tosave_path)
    print('saved ' + saved_path)
    return saved_path


def classes_df_dict_creation(logfile_path, classes_dict):
    clasiff_log = pd.read_csv(logfile_path)
    classes_data_dict = {}
    for one_class in classes_dict.keys():
        classes_data_dict[one_class] = []
    for total_class in classes_dict.keys():
        for index, classif_line in clasiff_log.iterrows():
            if classif_line['type'] == total_class:
                classes_data_dict[total_class].append(index)
    result_df_dict = {}
    for class_key in classes_data_dict.keys():
        result_df_dict[classes_dict[class_key]] = clasiff_log.loc[classes_data_dict[class_key]].reset_index(drop=True)
        print(class_key + ": " + str(len(result_df_dict[classes_dict[class_key]])))
    return result_df_dict


def create_report(classes_df_dict, files_path):
    FILES_processing_lib.create_floder('./results')
    report_doc = DOCX.DOCXwriter("./results/result_report.docx")
    report_doc.landscape = True
    report_doc.left_field = 0
    report_doc.right_field = 0
    report_doc.top_fieldd = 0
    report_doc.bottom_fieldd = 0
    report_doc.paragraph_spacing = 0
    report_doc.init()
    report_doc.normal_style()
    for key, df in zip(classes_df_dict.keys(), classes_df_dict.values()):
        print(key)
        report_doc.paragraph()
        report_doc.paragraph()
        report_doc.paragraph()
        report_doc.paragraph()
        report_doc.paragraph()
        report_doc.paragraph()
        report_doc.paragraph()
        report_doc.header(key)
        report_doc.paragraph()
        report_doc.page_break()
        total_counter = 1
        for _, file_data_row in df.iterrows():
            print(str(total_counter) + '/' + str(np.shape(df)[0]))
            total_counter = total_counter + 1
            file_name = file_data_row['file'].replace('\n', '')
            total_file_path = files_path + file_name
            saecg_img_path = create_saecg_img_from_data(total_file_path)
            report_doc.text_style.align = 'CENTER'
            report_doc.paragraph()
            report_doc.bold('Data for "'
                            + saecg_img_path.split('/')[-1].replace('.jpg', '') + '" (' + key + ')')
            report_doc.normal_style()
            report_doc.image(saecg_img_path, width=28)
            print('removing cache')
            os.remove(saecg_img_path)
            report_doc.page_break()
            report_doc.paragraph()
            report_doc.paragraph()
            report_doc.text_style.align = 'CENTER'
            report_doc.bold('Metadata')
            report_doc.normal_style()
            report_doc.paragraph()
            report_doc.normal('   ')
            file_metadata = file_data_row['metadata_json'].replace("'", '"')
            metadata_json_dict = dict(json.loads(file_metadata))
            for metadata_key in metadata_json_dict.keys():
                value = metadata_json_dict[metadata_key]
                if value == "":
                    report_doc.paragraph()
                    report_doc.bold(metadata_key + ": ")
                    report_doc.paragraph()
                else:
                    report_doc.bold(metadata_key + ": ")
                report_doc.normal(str(value) + ', ')
            report_doc.page_break()
    print('saving doc')
    report_doc.save()
    print('savinf pdf')
    saved_doc_path = report_doc.file_path
    PDF.word_to_pdf(saved_doc_path)
    print('done')


separated_df_dict = classes_df_dict_creation(logfile_path_0, classes_dict_0)
create_report(separated_df_dict, file_path_header_0)
