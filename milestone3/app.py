"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import numpy as np
import joblib
import pickle
from comet_ml import API


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

global_model = None

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    global global_model  # Declare the global variable

    path_default_model = r"log_reg_dist.sav"
    global_model = pickle.load(open(path_default_model, 'rb'))


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    try:
        # TODO: read the log file specified and return the data
        with open(LOG_FILE, 'r') as file:
            response = file.read()

        print(f"Response: {response}")

    except Exception as e:
        response = {"error": f"Error reading logs: {e}"}

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    global global_model  # Declare the global variable

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    workspace = json.get('workspace')
    model_name = json.get('model_name')
    version = json.get('version')


    if os.path.isfile(model_name):
        global_model = pickle.load(open(model_name, 'rb'))
        log_str = f"Success load model {model_name}"
        app.logger.info(log_str)
    else:
        try:
            # Download model
            api = API(api_key=os.environ.get('COMET_API_KEY'))
            models_dir = os.join.path("./")
            downloaded_model = api.get_model(workspace=workspace, model_name=model_name)
            downloaded_model.download(version, output_folder=models_dir, expand=True)

            # Load model
            path_model = os.path.join(models_dir, downloaded_model)
            global_model = pickle.load(open(path_model, 'rb'))
            
            log_str = f"Success download from Comet and load model {model_name}"
            app.logger.info(log_str)
        except:
            log_str = f"FAIL to download model from Comet"
            app.logger.info(log_str)

    response = log_str
    app.logger.info(response)
    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    try:
        global global_model  # Declare the global variable

        json = request.get_json()
        app.logger.info(json)

        X = []
        if "dist" in json:
            dist = json.get('dist')
            X.append(dist)
        if "angle" in json:
            angle = json.get('angle')
            X.append(angle)

        X = np.array([X])
        if len(X.shape) < 2:
            X = np.expand_dims(X, axis=0)

        y_pred = global_model.predict(X)[0]
        
        response = jsonify({'y_pred': int(y_pred)})
        app.logger.info(response)
        return response, 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500