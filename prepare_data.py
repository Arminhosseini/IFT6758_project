import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import comet_ml
from comet_ml import Experiment, Artifact




def main():
    """
    """

    # ================= Hyper-parameter ==================
    thaiv7_api_key = r"j3DrC3ChXkR42WfPCUh5EIkye"
    workspace_name = r"ift6758-b09-project"
    project_name = r"ift6758-project-milestone2"

    path_csv_fe2 = r"Dataset/tidyData_fe3.csv"
    train_season = [2016, 2017, 2018]
    val_season = [2019]
    test_season = [2020]

    path_train_csv = os.path.join("Dataset", "train.csv")
    path_val_csv = os.path.join("Dataset", "val.csv")
    path_test_csv = os.path.join("Dataset", "test.csv")

    # ====================================================


    # 1. Save feature engineering 2 to comet
    experiment = comet_ml.Experiment(api_key=thaiv7_api_key, project_name=project_name, workspace=workspace_name)

    artifact = Artifact(name="dataset", artifact_type="dataset", version="1.0.0")
    artifact.add(path_csv_fe2, metadata={"feature engineering 2": "dataframe"})
    experiment.log_artifact(artifact)

    experiment.end()
    print(f"Done store feature engineering 2 dataframe")

    # 2. Train-val-test split
    df = pd.read_csv(path_csv_fe2)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    df['season'] = df['gamePk'].apply(lambda x: int(str(x)[0:4]))

    train_set = df[df['season'].isin(train_season)]
    val_set = df[df['season'].isin(val_season)]
    test_set = df[df['season'].isin(test_season)]

    train_set.to_csv(path_train_csv)
    val_set.to_csv(path_val_csv)
    test_set.to_csv(path_test_csv)

    # 3. Save train-val-test to comet
    experiment = comet_ml.Experiment(api_key=thaiv7_api_key, project_name=project_name, workspace=workspace_name)
    artifact = Artifact(name="dataset", artifact_type="dataset", version="2.0.0")

    artifact.add(path_train_csv, metadata={"train set": f"after fe2, seasons = {train_season}"})
    artifact.add(path_val_csv, metadata={"val set": f"after fe2, seasons = {val_season}"})
    artifact.add(path_test_csv, metadata={"test set": f"after fe2, seasons = {test_season}"})

    experiment.log_artifact(artifact)
    experiment.end()
    print(f"Done store train-val-test")
        

main()