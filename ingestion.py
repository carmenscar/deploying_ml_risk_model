import pandas as pd
import numpy as np
import os
import json
import glob
#from datetime import datetime

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

def merge_multiple_dataframe(file_path_input, file_path_output):
    """
    Merges multiple CSV files from a specified input directory, removes duplicates, 
    and saves the resulting DataFrame to an output directory. Also creates a record 
    of the filenames read during the process and saves it in a text file.

    Parameters:
    file_path_input (str): Path to the folder containing the input CSV files.
    file_path_output (str): Path to the folder where the merged CSV and the 
                            record of read files will be saved.
    
    Side effects:
    - Saves a merged CSV file as 'finaldata.csv' in the specified output directory.
    - Creates a text file 'ingestedfiles.txt' in the output directory, containing 
      a list of the filenames of all the input CSV files that were read.
    """
    dataframes = []
    arquivos_lidos = []
    arquivos_csv = glob.glob(os.path.join(input_folder_path, "*.csv"))
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo, encoding="utf-8")
        dataframes.append(df)
        arquivos_lidos.append(os.path.basename(arquivo))
    df_merged = pd.concat(dataframes, ignore_index=True)
    df_merged.drop_duplicates(inplace=True)
    output_file_path = os.path.join(file_path_output, 'finaldata.csv')
    df_merged.to_csv(output_file_path, index=False, encoding="utf-8")
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as f:
        for arquivo in arquivos_lidos:
            f.write(f"{arquivo}\n")
    return df_merged

if __name__ == '__main__':
    df = merge_multiple_dataframe(input_folder_path, output_folder_path)
    print(df.head())

