from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
#app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

model_path_file = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
test_data_path = config['test_data_path']
model_path = config['prod_deployment_path']

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET'])
def predict():
    filename = request.args.get('filename') 
    csv_file_path = os.path.join(test_data_path, filename)
    predictions = model_predictions(model_path_file, csv_file_path)
    return jsonify(predictions.tolist()), 200

#######################Scoring Endpoint
@app.route("/scoring", methods=['POST','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    data = request.json
    filename = data.get('filename')
    model_name = data.get('model_name')
    f1_score_value = score_model(model_path, filename, model_name)
    return jsonify({"f1_score": f1_score_value}), 200

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    #return #return a list of all calculated summary statistics
    stats = dataframe_summary()
    return jsonify(stats), 200

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    na_values = missing_data()
    exec_time = execution_time()
    outdated_packs = outdated_packages_list()
    #return #add return value for all diagnostics
    diagnostics_results = {
        "missing_data_percentage": na_values,
        "execution_time": exec_time,
        "outdated_packages_list": outdated_packs }
    return jsonify(diagnostics_results), 200

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
    
