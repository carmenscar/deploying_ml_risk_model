#from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
ingest_path = os.path.join(config['output_folder_path'])
prod_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    model_file_path = os.path.join(model_path, 'trainedmodel.pkl')
    score_file_path = os.path.join(model_path, 'latestscore.txt')
    ingested_file_path = os.path.join(ingest_path, 'ingestedfiles.txt')

    if not os.path.exists(prod_path):
        os.makedirs(prod_path)

    shutil.copy(model_file_path, prod_path)
    shutil.copy(score_file_path, prod_path)
    shutil.copy(ingested_file_path, prod_path)

if __name__ == "__main__":
    store_model_into_pickle()
      
        
        

