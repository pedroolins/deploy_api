import pandas as pd
import sklearn
import numpy
from flask import Flask, request
from flask_basicauth import BasicAuth
import pickle
import os
from dotenv import load_dotenv

from wine_quality.Wine_Quality import WineQuality


## carrgando o modelo
model = pickle.load(open("model/modelo.pkl", "rb"))

## instanciando o flask
app = Flask(__name__)

## criando as variáveis de ambiente
load_dotenv()

## username
USER_NAME = os.getenv("USER_NAME")
##password
PASSWORD = os.getenv("PASSWORD")
print(USER_NAME)
print(PASSWORD)

## criando a nossa autenticação
app.config['BASIC_AUTH_USERNAME'] = USER_NAME
app.config['BASIC_AUTH_PASSWORD'] = PASSWORD

basic_auth = BasicAuth(app)

## criando meus endpoints
@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    ## coletando os dados
    test_json = request.get_json()
    if test_json:
        if isinstance(test_json, dict): #valor único
            df = pd.DataFrame(test_json, index=[0])
        else:
            df = pd.DataFrame(test_json)

    ## pre processing
    pipeline = WineQuality()
    df1 = pipeline.data_preparation(df)

    ## prediction
    pred = model.predict(df1)
    df1['prediction'] = pred
    response = df1

    return response.to_json(orient='records')

if __name__ == "__main__":
    ## start flask
    app.run(host="0.0.0.0", debug=True)
