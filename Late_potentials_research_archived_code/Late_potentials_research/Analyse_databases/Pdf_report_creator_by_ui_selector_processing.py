import Analyse_databases.class_UIselector
from Analyse_databases.modules import PDF
from Analyse_databases.modules import FILES_processing_lib
from Analyse_databases.modules import DOCX
import json
import numpy as np
from Analyse_databases.modules.WFDB import WfdbParce
from BIOMED_SIGNALS.ECG.class_record import EcgRecord
import pandas as pd
import os


# ------------------------------------Input variables-------------------------------------------#
logfile_path_0 = './ptb-diagnostic-ecg-database-1.0.0_logfile.csv'
file_path_header_0 = 'E:/Bases/PTB DATABASE/ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'
classes_dict_0 = {"LVP": 'Late Ventricular Potentials',
                  'LAP': 'Late Atrial Potentials',
                  'LAP_and_LVP': 'Late Ventricular and Atrial Potentials',
                  'no_LAP_no_LVP': 'No_Late_Potentials'}
# ----------------------------------------------------------------------------------------------#

base_name = logfile_path_0.split('/')[-1].replace('.csv','')
results_floder = './results/'+base_name+'/'
FILES_processing_lib.create_floder(results_floder)



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

def ui_img_generator(signals, leads, units, fs_mass, events, savepath):
    ui_selector = Analyse_databases.class_UIselector.UIselector()
    ui_selector.signals = signals
    ui_selector.leads = leads
    ui_selector.units = units
    ui_selector.fs_mass = fs_mass
    ui_selector.event_annotations = list(events)
    ui_selector.create_img_with_events(savepath)



def create_saecg_img_from_data(filepath, events_p):
    wfdb_file_object = WfdbParce(filepath).read()
    record_object_data = EcgRecord(Fs=wfdb_file_object.Fs,
                                   Signals=wfdb_file_object.Signals,
                                   Leads=wfdb_file_object.Leads,
                                   Units=wfdb_file_object.Units,
                                   Metadata=wfdb_file_object.Metadata)
    record_object_data = record_object_data.remove_leads('marker', 'abp', 'mcar', 'mcal', 'radi', 'thermst',
                                                         'flow_rate', 'o2', 'co2', 'PCG')  #
    last_record_len = list(record_object_data.ged_record_len().values())[-1]
    last_record_fs = record_object_data.Fs[-1]
    print('ECG fs: ' + str(last_record_fs) + ' len:' + str(last_record_len) + ' Sec (' + str(
        round(last_record_len / 60, 4)) + ' min)')
    record_object_data.saecg_count()
    print(record_object_data.Metadata)
    # signals_p = record_object_data.Signals
    signals_p = record_object_data.SAECG
    leads_p = record_object_data.Leads
    units_p = record_object_data.Units
    fs_mass_p = record_object_data.Fs
    path_to_save = './tmp_img.jpg'
    ui_img_generator(signals_p, leads_p, units_p, fs_mass_p, events_p, path_to_save)
    return path_to_save

def create_report(classes_df_dict, files_path):
    FILES_processing_lib.create_floder('./results')
    report_doc = DOCX.DOCXwriter(results_floder+base_name+".docx")
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
            events = file_data_row['events'].replace('\n', '')
            events = list(eval(events))
            total_file_path = files_path + file_name
            saecg_img_path = create_saecg_img_from_data(total_file_path, events)
            report_doc.text_style.align = 'CENTER'
            report_doc.paragraph()
            report_doc.bold('Data for "'
                            + total_file_path.split('/')[-1] + '" (' + key + ')')
            report_doc.normal_style()
            report_doc.image(saecg_img_path, width=28)
            print('removing cache')
            os.remove(saecg_img_path)
            report_doc.page_break()
            report_doc.paragraph()
            report_doc.paragraph()
            report_doc.text_style.align = 'CENTER'
            report_doc.bold('Late potentials Simson features')
            report_doc.normal_style()
            report_doc.paragraph()
            report_doc.text_style.align = 'LEFT'
            report_doc.bold('LAP features')
            LAP_simson_features = file_data_row['LAP_vmagnitude_features'].replace("'", '"')
            report_doc.normal(str(LAP_simson_features) + ', ')
            LVP_simson_features = file_data_row['LVP_vmagnitude_features'].replace("'", '"')
            report_doc.bold('LVP features')
            report_doc.normal(str(LVP_simson_features) + ', ')
            report_doc.text_style.align = 'CENTER'
            report_doc.paragraph()
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