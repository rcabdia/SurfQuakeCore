import os
from obspy import read
import pandas as pd

def list_folder_files(folder):
    list_of_files = []
    for top_dir, sub_dir, files in os.walk(folder):
        for file in files:
            list_of_files.append(os.path.join(top_dir, file))
    return list_of_files

def create_links(list_of_files, destination_folder):
    for file in list_of_files:
        path_symlink = os.path.basename(file)
        destination_file_link = os.path.join(destination_folder, path_symlink)
        os.symlink(file, destination_file_link)
        print(f"Symbolic link created from {file} to {destination_file_link}")
    print("End Creation list of folders")

cwd = os.getcwd()
path_to_files = os.path.join(cwd, "data")
output_to_files = os.path.join(cwd, "tmp")
list_of_files = list_folder_files(path_to_files)
create_links(list_of_files, output_to_files)
