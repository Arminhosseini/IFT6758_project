import json
import requests
import pandas as pd
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from new_crawler import Crawler
from process_new_dataset import *


def get_game_data_and_save_2_json(game_id, path_output_folder):
    """
    This function will get game data (by game_id) and save to json file
    * Argument:
    game_id -- a string, indicate the game id
    path_output_folder -- a string, indicate the path to output folder
    
    * Return:
    path_out_json_game_data -- a string, indicate path of output game data file (json file)
    """
    c = Crawler()
    json_game_data = c.get_game_data(game_id=game_id)

    if os.path.exists(path_output_folder) == False:
        os.mkdir(path_output_folder)

    path_out_json_game_data = os.path.join(path_output_folder, f"{game_id}.json")
    with open(path_out_json_game_data, "w") as file:
        json.dump(json_game_data, file)

    return path_out_json_game_data

    
def get_team_name(path_json_game_data):
    """
    This function will read game data (json file) and extract the name of home and away team
    * Arguments:
    path_json_game_data -- a string, indicate the path to json file
    
    * Returns:  
    home_team_name -- a string.
    away_team_name -- a string.
    """

    with open(path_json_game_data) as json_file:
        game_data = json.load(json_file)

    home_team_name = game_data['homeTeam']['name']['default']
    away_team_name = game_data['awayTeam']['name']['default']

    return (home_team_name, away_team_name)


def get_actual_goal(path_json_game_data):
    """
    This function read the game data (json file) and get the actual goal
    * Arguments:
    path_json_game_data -- a string, indicate the path to json file
    
    * Returns:    
    goals -- a dictionay, with the following format {'homeScore': 0, 'awayScore': 0}
    """

    goals = {'homeScore': 0, 'awayScore': 0}

    with open(path_json_game_data) as json_file:
        game_data = json.load(json_file)

    # Loop through each event of games
    list_event = game_data['plays'] 
    for event in list_event:
        event_type = event['typeDescKey']
        if event_type == "goal":
            goals['homeScore'] = event['details']['homeScore']
            goals['awayScore'] = event['details']['awayScore']
    
    return goals


def process_feature(path_json_game_data):
    """
    This function will read game data (json file) and return processed dataframe
    * Argument:
    path_json_game_data -- a string, indicate the path to json file

    * Returns:
    game_df -- a data frame, indicate the output dataframe
    """

    game_df = get_list_event_of_game(path_json_game_data)

    # Process more feature
    game_df['shot_distance'] = game_df.apply(compute_shot_distance, axis=1)
    game_df['shot_angle'] = game_df.apply(compute_shot_angle, axis=1)
    game_df['isgoal'] = game_df['event type'].apply(lambda x: 1 if x == 'goal' else 0)
    game_df['is empty net'] = game_df.apply(check_empty_net, axis=1)

    if "Unnamed: 0" in game_df.columns:    # Remove redundant features
        game_df = game_df.drop(columns=['Unnamed: 0'])

    return game_df


def get_last_event(path_json_game_data):
    """
    This function will read game data (json file) and get the last event of the game 

    * Argument:
    path_json_game_data -- a string, indicate the path to json file

    * Returns:
    last_event -- a dictionary, indicate the info of last event
    """

    with open(path_json_game_data) as json_file:
        game_data = json.load(json_file)

    # Extract info of last event
    list_events = list(game_data['plays'])
    last_event = list_events[-1]

    return last_event



# if __name__ == "__main__":

#     # ============================================================
#     path_output_folder = os.path.join("game_client_data")
#     game_id = r"2017030111"

#     features = ['shot_distance']
#     # ============================================================

#     file_predict_name = f"{game_id}_{''.join(features)}.csv"
#     path_output_file_predict = os.path.join(path_output_folder, file_predict_name)

#     # Check if we already predict it
#     if os.path.exists(path_output_file_predict):  
#         game_df_pred = pd.read_csv(path_output_file_predict)

#     else:
#         path_out_json_game_data = get_game_data_and_save_2_json(game_id, path_output_folder)

#         game_df = process_feature(path_out_json_game_data)

#         # Query prediction
#         serving_client = ServingClient(features=features)
#         list_output = serving_client.predict(game_df)

#         # Store prediction to file
#         game_df_pred = pd.concat([game_df, list_output], axis=1)
#         game_df_pred.to_csv(path_output_file_predict)