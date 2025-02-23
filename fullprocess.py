
#import diagnostics
#import reporting
import json
import os
import glob

from training import train_model
from scoring import score_model
from deployment import store_model_into_pickle
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list


##################Check and read new data
#first, read ingestedfiles.txt
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
model_path_ws = config['output_model_path']
test_data_path = config['test_data_path']

ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
with open(ingested_files_path, 'r') as f:
    ingested_files = set(f.read().splitlines())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
current_files = set(os.path.basename(f) for f in glob.glob(os.path.join(input_folder_path, '*.csv')))
new_files = current_files - ingested_files
print(new_files)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    print("Nenhum novo arquivo encontrado. Processo encerrado.")
    exit()
else:
    print(f"Novos arquivos encontrados: {new_files}. Prosseguindo com a ingestão.")
    os.system('python ingestion.py')

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
# Ler a pontuação do modelo atual
latest_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(latest_score_path, 'r') as f:
    current_score = float(f.read().strip())

print("Treinando um novo modelo...")
train_model(output_folder_path, model_path_ws)

print("Calculando a pontuação do novo modelo...")
test_path = 'testdata.csv'
model_trained  = 'trainedmodel.pkl'
new_score = score_model(model_path_ws, test_path, model_trained)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_score < current_score:
    print(f"Model drift detectado. Pontuação atual: {current_score}, Nova pontuação: {new_score}.")
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    print("Re-deployment...")
    os.system('python deployment.py')
    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    print("New diagnoses...")
    os.system('python diagnostics.py')
    print("New reports...")
    os.system('python reporting.py')
else:
    print("there is no model drift")







