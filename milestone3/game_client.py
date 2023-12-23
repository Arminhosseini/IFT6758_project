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


def get_game_data_json(game_id):
    """
    This function will get game data (by game_id)
    * Argument:
    game_id -- a string, indicate the game id
    path_output_folder -- a string, indicate the path to output folder
    
    * Return:
    json_game_data -- a json dictionary, indicate the game data under json format
    """
    c = Crawler()
    json_game_data = c.get_game_data(game_id=game_id)
    return json_game_data

    
def get_team_name(json_game_data):
    """
    This function extract the name of home and away team
    * Arguments:
    json_game_data -- a json dictionary, indicate the game data under json format
    
    * Returns:  
    home_team_name -- a string.
    away_team_name -- a string.
    """
    try:
        home_team_name = json_game_data['homeTeam']['name']['default']
        away_team_name = json_game_data['awayTeam']['name']['default']
    except:
        home_team_name = away_team_name = None

    return (home_team_name, away_team_name)


def get_actual_goal(json_game_data):
    """
    This function get the actual goal
    * Arguments:
    json_game_data -- a json dictionary, indicate the game data under json format
    
    * Returns:    
    goals -- a dictionay, with the following format {'homeScore': 0, 'awayScore': 0}
    """

    goals = {'homeScore': 0, 'awayScore': 0}

    # Loop through each event of games
    list_event = json_game_data['plays'] 
    for event in list_event:
        event_type = event['typeDescKey']
        if event_type == "goal":
            goals['homeScore'] = event['details']['homeScore']
            goals['awayScore'] = event['details']['awayScore']
    
    return goals


def process_feature(json_game_data):
    """
    This function will read game data (json file) and return processed dataframe
    
    * Arguments:
    json_game_data -- a json dictionary, indicate the game data under json format

    * Returns:
    game_df -- a data frame, indicate the output dataframe
    """

    game_df = get_list_event_of_game(json_game_data)

    # Process more feature
    game_df['shot_distance'] = game_df.apply(compute_shot_distance, axis=1)
    game_df['shot_angle'] = game_df.apply(compute_shot_angle, axis=1)
    game_df['isgoal'] = game_df['event type'].apply(lambda x: 1 if x == 'goal' else 0)
    game_df['is empty net'] = game_df.apply(check_empty_net, axis=1)

    game_df = game_df.loc[:, ~game_df.columns.str.contains('^Unnamed')]

    return game_df


def get_last_event(json_game_data):
    """
    This function will read game data (json file) and get the last event of the game 

    * Arguments:
    json_game_data -- a json dictionary, indicate the game data under json format

    * Returns:
    last_event -- a dictionary, indicate the info of last event
    """

    # Extract info of last event
    list_events = list(json_game_data['plays'])
    last_event = list_events[-1]

    return last_event


def handle_saved_game_id(previous_game_df, current_game_df, serving_client):
    
    if len(previous_game_df) == len(current_game_df): # nothing news happend
        return previous_game_df
    
    elif len(previous_game_df) < len(current_game_df): # new data is coming
        try:
            # Extract new event
            n_old_samples = previous_game_df.shape[0]
            n_new_samples = current_game_df.shape[0]
            # game_df_new = current_game_df.iloc[n_old_samples:n_new_samples, :]
            game_df_new = current_game_df.tail(n_new_samples - n_old_samples)

            # Perform prediction
            list_output_new = serving_client.predict(game_df_new)
            game_df_new_pred = pd.concat([game_df_new, list_output_new], axis=1, ignore_index=True)

            # Merge old prediction and new prediction
            previous_game_df = previous_game_df.loc[:, ~previous_game_df.columns.str.contains('^Unnamed')]
            game_df_new_pred = game_df_new_pred.loc[:, ~game_df_new_pred.columns.str.contains('^Unnamed')]

            game_df_pred = pd.concat([previous_game_df, game_df_new_pred], axis=0, ignore_index=True)
            return game_df_pred
        except:
            # previous_game_df = previous_game_df.loc[:, ~previous_game_df.columns.str.contains('^Unnamed')]
            return None

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