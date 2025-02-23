import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
##############Function for reporting
def score_model(test_data, model_path):
    """
    calculate a confusion matrix(cm) using the test data and the deployed model and save 
    cm pic
    """
    test_data_path = os.path.join(test_data, 'testdata.csv')
    test_data = pd.read_csv(test_data_path, encoding="utf-8")

    model_file_path = os.path.join(model_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as m:
        model = pickle.load(m)

    X_test = test_data.drop(columns=['corporation','exited']).values
    y_test = test_data['exited'].values.ravel()
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsões')
    plt.ylabel('Valores Reais')
    confusion_matrix_path = os.path.join(model_path, 'confusionmatrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()


if __name__ == '__main__':
    score_model(test_data_path,output_model_path)
