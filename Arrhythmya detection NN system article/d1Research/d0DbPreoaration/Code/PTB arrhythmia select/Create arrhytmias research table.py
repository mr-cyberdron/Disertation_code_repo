import pandas as pd
from ClassificationScheme import arrhythmias_group, arrhythmia_warnings_group, normal_ecg_group
from ClassificationScheme import arrhythmias_group_names, arrhythmia_warnings_group_names, normal_ecg_group_names

dbPath = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
metadataPath = dbPath+'ptbxl_database.csv'

metadataDb = pd.read_csv(metadataPath)

new_metadata_db_dict = {'fpath_from_db_ROOT':[],'age':[],'sex':[], 'weight':[],
                        'scp_codes':[],'validated_by_doctor':[],'target SCP':[],'group':[],
                        'subgroup':[],
                        'scp_descryption':[],'doctor_report':[]}

general_groups = ['arrhythmias','arrhythmia_warnings', 'ecg_in_norm']
general_groups_containers = [arrhythmias_group, arrhythmia_warnings_group, normal_ecg_group]
subgroups_names = [arrhythmias_group_names, arrhythmia_warnings_group_names, normal_ecg_group_names]
for group_container,sub_names, group_name in zip(general_groups_containers,subgroups_names,general_groups):
    for subgroup, subgroup_name in zip(group_container,sub_names):
        for scp_dict in subgroup:
            scp_code = list(scp_dict.keys())[0]
            scp_description = scp_dict[scp_code]
            scp_code_db_slice = metadataDb[
                metadataDb['scp_codes'].str.contains("'" + scp_code + "'", na=False, regex=False)]
            for i, row in scp_code_db_slice.iterrows():
                fpath_from_root = row['filename_hr']

                new_metadata_db_dict['fpath_from_db_ROOT'].append(fpath_from_root)
                new_metadata_db_dict['age'].append(row['age'])
                new_metadata_db_dict['sex'].append(row['sex'])
                new_metadata_db_dict['weight'].append(row['weight'])
                new_metadata_db_dict['scp_codes'].append(row['scp_codes'])
                new_metadata_db_dict['validated_by_doctor'].append(row['validated_by_human'])
                new_metadata_db_dict['target SCP'].append(scp_code)
                new_metadata_db_dict['group'].append(group_name)
                new_metadata_db_dict['subgroup'].append(subgroup_name)
                new_metadata_db_dict['scp_descryption'].append(scp_description)
                new_metadata_db_dict['doctor_report'].append(row['report'])

generated_db = pd.DataFrame(data=new_metadata_db_dict)
generated_db.to_csv('arrhytmia_groups_metadata.csv')

generated_db_arrhythmias = generated_db[generated_db['group'].str.contains('arrhythmias', na=False, regex=False)]
generated_db_arrhythmias_warnings = generated_db[generated_db['group'].str.contains('arrhythmia_warnings', na=False, regex=False)]
generated_db_ecg_in_norm = generated_db[generated_db['group'].str.contains('ecg_in_norm', na=False, regex=False)]

arrhythmias_path_mass = generated_db_arrhythmias['fpath_from_db_ROOT'].to_numpy()
arrhythmias_path_scp_mass = generated_db_arrhythmias['scp_codes'].to_numpy()
arrhythmias_warnings_path_mass = generated_db_arrhythmias_warnings['fpath_from_db_ROOT'].to_numpy()
arrhythmias_warnings_scp_mass = generated_db_arrhythmias_warnings['scp_codes'].to_numpy()

only_warnings_path_mass = []
for path1, scp_codes in zip(arrhythmias_warnings_path_mass, arrhythmias_warnings_scp_mass):
    if path1 not in arrhythmias_path_mass:
        only_warnings_path_mass.append(path1)


only_warnings_db = generated_db_arrhythmias_warnings[generated_db_arrhythmias_warnings['fpath_from_db_ROOT'].isin(only_warnings_path_mass)]

print(len(generated_db_arrhythmias))
print(len(generated_db_arrhythmias_warnings))
print(len(generated_db_ecg_in_norm))
print(len(only_warnings_db))











