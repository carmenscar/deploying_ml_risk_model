import requests
import os
import json

URL = "http://127.0.0.1:8000"

# Nome do arquivo CSV
filename = "testdata.csv"
response1 = requests.get(f'{URL}/prediction', params={'filename': filename})

data = {
    "filename": "testdata.csv",
    "model_name": "trainedmodel.pkl"
}
response2 = requests.post(f'{URL}/scoring', json=data)

response3 = requests.get(f'{URL}/summarystats')

response4 = requests.get(f'{URL}/diagnostics')

responses = {
    'prediction': response1.json(),
    'scoring': response2.text,
    'summarystats': response3.json(),
    'diagnostics': response4.json()
}
#write the responses to your workspace
with open('api_responses.json', 'w') as f:
    json.dump(responses, f)

print(responses)


