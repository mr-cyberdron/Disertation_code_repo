from sys import platform
import os


def check_for_special_symbol(text: str):
    special_characters = ",<\>"
    for c in text:
        if c in special_characters:
            raise TypeError(f"Special character '{c}' forbidden! Path format should be: Path/to/ile.format")
        else:
            pass


def word_to_pdf(word_file_path):
    """
            Also formats odf, odt
            Libreoffice must be installed: libreoffice.org
    """
    check_for_special_symbol(word_file_path)
    file_name = word_file_path.split("/")[-1]
    outdir = word_file_path.replace(file_name, '')
    if platform == "linux" or platform == "linux2":
        cmd_comand = f'lowriter --headless --convert-to pdf "{word_file_path}" --outdir "{outdir}"'
        cmd_comand = f'cmd /c "{cmd_comand}"'
        print(cmd_comand)
        os.system(cmd_comand)
    elif platform == "win32":
        cmd_comand = f'"C:/Program Files/LibreOffice/program/swriter.exe" ' \
                     f'--headless --convert-to pdf "{word_file_path}" --outdir "{outdir}"'
        cmd_comand = f'cmd /c "{cmd_comand}"'
        print(cmd_comand)
        os.system(cmd_comand)
