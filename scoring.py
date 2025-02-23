#from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_file_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path'])
#################Function for model scoring
def score_model(model_path, test_path, model):
    """
    this function should take a trained model, load test data,
     and calculate an F1 score for the model relative to the test data
    """
    model_file_path = os.path.join(model_path, model)
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    test_file_path = os.path.join(test_data_path, test_path)
    test_data = pd.read_csv(test_file_path, encoding="utf-8")
    X_test = test_data.drop(columns=['corporation','exited']).values
    y_test = test_data['exited'].values.ravel()
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    score_file_path = os.path.join(model_path, 'latestscore.txt')
    with open(score_file_path, 'w') as f:
        f.write(str(f1))
    return f1
    #it should write the result to the latestscore.txt file

if __name__ == '__main__':
    test_path = 'testdata.csv'
    model_trained  = 'trainedmodel.pkl'
    test = score_model(model_file_path, test_path, model_trained)
    print(test)
