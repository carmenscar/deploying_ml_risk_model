
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path_file = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')

##################Function to get model predictions
def model_predictions(model_path, test_path):
    """
    read the deployed model and a test dataset, calculate predictions
    """
    with open(model_path, 'rb') as m:
        model = pickle.load(m)

    test_path_csv = os.path.join(test_path, 'testdata.csv')
    df_test = pd.read_csv(test_path_csv, encoding="utf-8")

    X_test = df_test.drop(columns=['corporation','exited']).values
    y_pred = model.predict(X_test)

    return y_pred

##################Function to get summary statistics
def dataframe_summary(dataset_path):
    """
    This function calculates mean, median, and standard deviation for each numeric column in the dataset.
    :return: A list of dictionaries containing summary statistics
    """
    df_file_path = os.path.join(dataset_path, 'finaldata.csv')
    df = pd.read_csv(df_file_path, encoding="utf-8")
    
    summary = {
        "column": [],
        "mean": [],
        "median": [],
        "std_dev": []
    }
    
    for col in df.select_dtypes(include=[np.number]).columns:
        summary["column"].append(col)
        summary["mean"].append(df[col].mean())
        summary["median"].append(df[col].median())
        summary["std_dev"].append(df[col].std())
    
    return [summary]

def execution_time():
    """
    #calculate timing of training.py and ingestion.py
    """
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    os.system('python training.py')
    timing=timeit.default_timer() - starttime
    return print(timing)

def missing_data(dataset_path):
    """
    This function checks for missing (NA) data in the dataset and calculates the percentage of missing values per column.
    :return: A list containing the percentage of NA values for each column
    """
    df_file_path = os.path.join(dataset_path, 'finaldata.csv')
    df = pd.read_csv(df_file_path, encoding="utf-8")
    missing_percentage = (df.isna().sum() / len(df)) * 100
    print(missing_percentage)
    
    return missing_percentage.tolist()

##################Function to check dependencies
def outdated_packages_list():
    # Verificar pacotes desatualizados
    outdated = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf-8')
    for line in outdated.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 3:
            package = parts[0]
            installed_version = parts[1]
            latest_version = parts[2]
            print(f"package | installed_version | latest_version")
            print(f"{package} | {installed_version} | {latest_version}")

if __name__ == '__main__':
    list_y_pred = model_predictions(model_path_file,test_data_path)
    print(list_y_pred)
    summary = dataframe_summary(dataset_csv_path)
    print(summary)
    execution_time()
    missing_data(dataset_csv_path)
    outdated_packages_list()





    
