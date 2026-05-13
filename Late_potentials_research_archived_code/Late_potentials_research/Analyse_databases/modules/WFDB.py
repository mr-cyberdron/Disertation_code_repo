import numpy as np
import wfdb
import copy


class WfdbParce:
    """
    Parsing Wfdb files:
    Fs, Signals, Leads, Units, Metadata
    example:
        heafile_location = 'file_header.hea'
        wfdb_file_object = WfdbParce(heafile_location).read()
        print(wfdb_file_object.Fs)
    """

    def __init__(self, wfdb_hea_path):
        print('WFDB parser init')
        self.Hea_path = wfdb_hea_path
        self.Wfdb_content = None

        self.Fs = None
        self.Signals = None
        self.Leads = None
        self.Units = None
        self.Metadata = None

    @staticmethod
    def read_fs(content, leads_num) -> list[int]:
        fs = content[1]['fs']
        if type(fs) is int:
            fs_mass = []
            for leads_counter in range(leads_num):
                fs_mass.append(fs)
        else:
            fs_mass = list(fs)
        return fs_mass

    @staticmethod
    def read_signals(content) -> np.ndarray:
        sig_content = content[0].transpose()
        return sig_content

    @staticmethod
    def read_leads(content) -> list[str]:
        leads_list = content[1]['sig_name']
        return list(leads_list)

    @staticmethod
    def read_units(content) -> list[str]:
        units_list = content[1]['units']
        return list(units_list)

    @staticmethod
    def read_metadata(content) -> dict[str:str]:
        metadata_list = content[1]['comments']
        metadata_dict = {}
        for metadata_item in metadata_list:
            item_split = metadata_item.split(':')
            item_key = item_split[0]
            item_split.remove(item_key)
            item_split.append('')
            item_value = ''.join(item_split)
            metadata_dict[item_key] = item_value
        return metadata_dict

    def read(self):
        new_hea_path = copy.deepcopy(self.Hea_path).replace('.hea', '')
        wfdb_content = wfdb.rdsamp(new_hea_path)
        self.Wfdb_content = wfdb_content
        self.Signals = self.read_signals(wfdb_content)
        self.Fs = self.read_fs(wfdb_content, len(self.Signals))
        self.Leads = self.read_leads(wfdb_content)
        self.Units = self.read_units(wfdb_content)
        self.Metadata = self.read_metadata(wfdb_content)
        return self
