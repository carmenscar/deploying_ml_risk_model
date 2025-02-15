from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path_ws = os.path.join(config['output_model_path']) 

#################Function for training the model
def train_model(dataset_path, model_path):
    """
    Trains a logistic regression model using a CSV dataset and saves the trained model to a file.

    Parameters:
    dataset_path (str): Path to the directory where the 'finaldata.csv' file is located.
    model_path (str): Path to the directory where the trained model will be saved.

    Returns:
    None
    """
    output_file_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(output_file_path, encoding="utf-8")

    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    #fit the logistic regression to your data
    X_train = df.drop(columns=['corporation','exited']).values
    y_train = df['exited'].values.ravel()
    model = lr_model.fit(X_train, y_train)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    model_file_path = os.path.join(model_path, 'trainedmodel.pkl')
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

    return None

if __name__ == "__main__":
    train_model(dataset_csv_path, model_path_ws)

