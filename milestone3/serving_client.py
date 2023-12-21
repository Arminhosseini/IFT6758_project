import json
import requests
import pandas as pd
import logging
import os
from pathlib import Path
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):

        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["shot_distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        list_features = self.features
        list_input = X[list_features]
        list_output = []

        for _, input in list_input.iterrows():
            
            # Get input features
            input_features = {}
            for feature in list_features:
                input_features[feature] = input[feature]

            # Send request to server
            response = requests.post(f"{self.base_url}/predict", json=input_features)
            if response.status_code == 200:
                result_json = response.json()
                y_pred = result_json['y_pred']
                list_output.append(y_pred)
            else:    
                y_pred = -1.0
                list_output.append(y_pred)

        return pd.DataFrame({"y_pred": list_output})


    def logs(self) -> dict:
        """Get server logs"""

        response = requests.get(f"{self.base_url}/logs")

        if response.status_code == 200:
            logs_data = response.json()
            return logs_data
        else:
            response.raise_for_status()

    def download_registry_model(self, workspace: str, model_name: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        input_json = {'workspace': workspace, 'model_name': model_name, 'version': version}
    
        response = requests.post(f"{self.base_url}/download_registry_model", json=input_json)
        if response.status_code == 200:
            download_info = response.json()
            return download_info
        else:    
            response.raise_for_status()
    


# if __name__ == "__main__":
    
#     load_dotenv()
#     PATH_INPUT_DATAFRAME = Path(r"/home/thaiv7/Desktop/IFT6758_project/dataset_new/processed_data.csv")

#     X = pd.read_csv(PATH_INPUT_DATAFRAME)

#     feature = ['shot_distance']
#     serving_client = ServingClient(features=feature)

#     # list_output = serving_client.predict(X)
#     # logs_data = serving_client.logs()

#     workspace = r"ift6758-b09-project"
#     model_name="log_reg_ang"
#     version = "1.0.0"

#     download_info = serving_client.download_registry_model(workspace, model_name, version)
