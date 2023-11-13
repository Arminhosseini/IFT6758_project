import pandas as pd
import os
import comet_ml
from dotenv import load_dotenv
load_dotenv()


PATH_TIDY_DATA_CSV = r"Dataset/feature_engineering2.csv"
df = pd.read_csv(PATH_TIDY_DATA_CSV)
game = df[df['gamePk'] == 2017021065]

workspace_name = r"ift6758-b09-project"
project_name = r"ift6758-project-milestone2"
slzhou_api_key = os.environ.get('COMET_API_KEY')

experiment = comet_ml.Experiment(api_key=slzhou_api_key, project_name=project_name, workspace=workspace_name)

experiment.log_dataframe_profile(game, name='wpg_v_wsh_2017021065', dataframe_format='csv')

experiment.end()
