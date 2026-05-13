import sys
import pandas as pd
import json
import os
import colorama


def show_img(path):
    os.system("'gwenview' " + path)


def show_json(filepath):
    os.system('firefox "' + filepath + '"')


def show_json_from_var(var: dict):
    with open('/tmp/tmp.json', 'w') as f:
        json.dump(var, f)
    f.close()
    os.system('firefox "' + '/tmp/tmp.json' + '"')


def show_df_in_csv(df, windows=False):
    def save_df(dff, p=os.path.expanduser('~'), n="tmp"):
        name = p + "/" + str(n) + ".csv"
        dff.to_csv(name, index=True)
        print("saved:" + name)
        return name

    if not windows:
        file = save_df(df, p=os.path.expanduser('~'), n="tmp")
        print("Showing")
        os.system('libreoffice --calc' + ' "' + file + '"')
    else:
        file = save_df(df, p='c:/tmp', n="tmp")
        print("Showing")
        out = '"C:\Program Files\LibreOffice\program\scalc.exe"' + ' ' + file
        print(out)
        os.system(out)


def show_numpy_in_csv(np_array):
    df = pd.DataFrame(data=np_array)
    show_df_in_csv(df)


def txt_log(text, p='./LOG.txt'):
    with open(p , 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()


def txt_log_read(p='./LOG.txt') -> list:
    with open(p, 'r') as f:
        text = f.readlines()
        text = [line.rstrip("\n") for line in text]
    f.close()
    return text


def print2(line):
    sys.stdout.write("\r{}".format(line))
    sys.stdout.flush()


class ColorPrint:
    colorama.init(autoreset=True)

    def __init__(self, text):
        self.text = text

    def red(self):
        print(colorama.Fore.LIGHTRED_EX + self.text)

    def green(self):
        print(colorama.Fore.LIGHTGREEN_EX + self.text)

    def blue(self):
        print(colorama.Fore.LIGHTBLUE_EX + self.text)

    def magneta(self):
        print(colorama.Fore.LIGHTMAGENTA_EX + self.text)

    def cyan(self):
        print(colorama.Fore.LIGHTCYAN_EX + self.text)

    def yellow(self):
        print(colorama.Fore.YELLOW + self.text)

    def bgred(self):
        print(colorama.Back.LIGHTRED_EX + colorama.Fore.BLACK + self.text)

    def bggreen(self):
        print(colorama.Back.LIGHTGREEN_EX + colorama.Fore.BLACK + self.text)

    def default(self):
        print(self.text)
