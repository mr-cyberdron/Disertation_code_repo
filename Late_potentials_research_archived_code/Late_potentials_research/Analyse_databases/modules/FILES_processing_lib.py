import os
import json
from pathlib import Path
import numpy as np


def scandir(dir_path, ext=None):
    """
    Returns the list of included files
    two ways of use (example):
    scandir(path,'.edf') - finds all files with .edf extension
    scandir(path,['.edf', 'txt']) - finds all files with .edf and .txt extension
    """
    def files_list(all_files, extension):
        res = []
        for file in all_files:
            if file.endswith(extension):
                res.append(file)
        return res
    listed_dir = os.listdir(dir_path)
    if type(ext) is str:
        listed_dir = files_list(listed_dir, ext)
    elif type(ext) is list:
        res2 = []
        for inst in ext:
            res2.append(np.array(files_list(listed_dir, inst)))
        to_concat_lists = tuple(res2)
        listed_dir = np.concatenate(to_concat_lists).tolist()
    return listed_dir


def save_json(dictt: dict, outpath: str):
    with open(outpath, 'w') as outfile:
        json.dump(dictt, outfile)


def create_floder(path):
    if not os.path.exists(path):
        os.makedirs(path)
