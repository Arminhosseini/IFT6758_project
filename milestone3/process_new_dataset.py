import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def define_net_coordinate(game_df):
    """
    This function will find the coordinate of the net in one game.

    game_df: dataframe of current game
    """

    dict_net_coordinate = {}

    list_period = game_df['period'].unique()
    for period in list_period:
        dict_net_coordinate[period] = {}

        for team in ['home', 'away']:
            try:
                current_df = game_df[(game_df['period'] == period) & (game_df['zone code'] == "O") & (game_df['team'] == team)].sample(1)
                current_x_coor = current_df['x coor'].iloc[0]    # extract x coordinate
                if int(current_x_coor) <= 0:
                    net_coordinate_x = -89
                    net_coordinate_y = 0
                elif int(current_x_coor) >= 0:
                    net_coordinate_x = 89
                    net_coordinate_y = 0
                dict_net_coordinate[period][team] = (net_coordinate_x, net_coordinate_y)
            except:
                dict_net_coordinate[period][team] = -1
    
    # Fill in -1 value (if has)
    for period in list_period:
        for team in ['home', 'away']:
            if dict_net_coordinate[period][team] == -1:  # There is missing value
                try:
                    if team == 'home':
                        net_coordinate_x = -1 * dict_net_coordinate[period]['away'][0]
                        net_coordinate_y = 0
                        dict_net_coordinate[period][team] = (net_coordinate_x, net_coordinate_y)
                    elif team == 'away':
                        net_coordinate_x = -1 * dict_net_coordinate[period]['home'][0]
                        net_coordinate_y = 0
                        dict_net_coordinate[period][team] = (net_coordinate_x, net_coordinate_y)
                except:   
                    dict_net_coordinate[period][team] = (np.nan, np.nan)
    return dict_net_coordinate



def get_list_event_of_game(path_game_file, list_chosen_event = ['shot-on-goal', 'goal']):
    """
    This function get list event of the game.

    * Argument:
    path_game_file -- a string, indicate path to the json file
    list_chosen_event -- a list of string

    * Return:
    game_df -- a dataframe, indicate the dataframe of list events
    """

    with open(path_game_file) as json_file:
        game_data = json.load(json_file)

    list_event_type = []
    list_x_coor = []
    list_y_coor = []
    list_event_owner_team_id = []
    list_zone_code = []
    list_period = []
    list_team = []
    list_situation_code = []

    # Extract all event of game
    home_team_id = game_data['homeTeam']['id']
    away_team_id = game_data['awayTeam']['id']
    list_event = game_data['plays'] 

    # Loop through each event of game
    for event in list_event:
        try:
            event_type = event['typeDescKey']
            if event_type in list_chosen_event: # Just choose shot event
                x_coor = event['details']['xCoord']
                y_coor = event['details']['yCoord']

                event_owner_team_id = event['details']['eventOwnerTeamId']

                if str(event_owner_team_id) == str(home_team_id):
                    team = 'home'
                elif str(event_owner_team_id) == str(away_team_id):
                    team = "away"
                else:
                    team = ""

                zone_code = event['details']['zoneCode']
                period = event['period']
                situation_code = event['situationCode']

                list_event_type.append(event_type)
                list_x_coor.append(x_coor)
                list_y_coor.append(y_coor)
                list_event_owner_team_id.append(event_owner_team_id)
                list_team.append(team)
                list_zone_code.append(zone_code)
                list_period.append(period)
                list_situation_code.append(situation_code)
        except:
            continue


    game_df = {"event type": list_event_type,\
                "period": list_period,\
                "x coor": list_x_coor,\
                "y coor": list_y_coor,\
                "owner team id": list_event_owner_team_id,\
                "team": list_team,\
                "zone code": list_zone_code,\
                "situation code": list_situation_code}
    game_df = pd.DataFrame(game_df)

    # Extract coordinate (x,y) of the net
    dict_net_coor = define_net_coordinate(game_df)

    list_net_x_coor = []
    list_net_y_coor = []
    for _, row in game_df.iterrows():
        period = row['period']
        team = row['team']
        net_coor_x, net_coor_y = dict_net_coor[period][team]
        list_net_x_coor.append(net_coor_x)
        list_net_y_coor.append(net_coor_y)
    game_df['net x coor'] = list_net_x_coor
    game_df['net y coor'] = list_net_y_coor

    return game_df


def compute_shot_distance(row):
    """
    This function calculate the distance from shot to net.

    * Arguments:
    row -- a series, indicate the information of 1 event.

    * Returns:
    distance -- a float number.
    """

    x = float(row['x coor'])
    y = float(row['y coor'])
    net_x = float(row['net x coor'])
    net_y = float(row['net y coor'])

    distance = np.nan
    try:  distance = np.sqrt((x - net_x) ** 2 + (y - net_y) ** 2)
    except:  distance = np.nan

    return distance


def compute_shot_angle(row):
    """
    This function calculate the angle of the shot.

    * Arguments:
    row -- a series, indicate the information of 1 event.

    * Returns:
    shot_angle_rad -- a float number, indicate the angle in radian unit.
    """

    try:
        shot_angle_rad = np.arcsin(row['y coor'] / row['shot_distance'])
    except:
        shot_angle_rad = np.nan
    return shot_angle_rad


def check_empty_net(row):
    """
    This function will check whether the net is empty or not.

    * Arguments:
    row -- a series, indicate the information of 1 event.

    * Returns:
    is_empty -- a boolen. 
                If is_empty = 0: not empty
                If is_empty = 1: empty
    """

    is_empty = None

    situation_code = str(row['situation code'])
    team = str(row['team'])
    if team == "home":
        away_goalie = str(situation_code[0])  # If home shot, we need to check goalie of away
        if away_goalie == '1':    
            is_empty = 0
        elif away_goalie == '0':
            is_empty = 1
        else:
            is_empty = np.nan
    elif team == "away":
        home_goalie = str(situation_code[3])  # if away shot, we need to check goalie of home
        if home_goalie == '1':  
            is_empty = 0
        elif home_goalie == '0':
            is_empty = 1
    else:
        is_empty = 0
    
    return is_empty


if __name__ == "__main__":

    # ======================================================================================
    PATH_FOLDER_DATA = Path(r"/home/thaiv7/Desktop/IFT6758_project/dataset_new")
    PATH_OUTPUT_FILE = Path(r"/home/thaiv7/Desktop/IFT6758_project/dataset_new/processed_data.csv")

    LIST_SEASON = [2016, 2017, 2018, 2019, 2020]
    LIST_GAME_TYPE = ['playoffs', 'regular_season']
    LIST_CHOSEN_EVENT = ['shot-on-goal', 'goal']

    # ======================================================================================

    df = pd.DataFrame()    # Final data frame

    for season in LIST_SEASON:
        path_season_folder = os.path.join(PATH_FOLDER_DATA, str(season))
        for game_type in LIST_GAME_TYPE:

            # Get list game (list of json file)
            path_season_game_folder = os.path.join(path_season_folder, game_type)
            list_game_name = sorted(os.listdir(path_season_game_folder))

            # Loop through all game in season
            for game_name in list_game_name:
                path_game_file = os.path.join(path_season_game_folder, game_name)
                game_df = get_list_event_of_game(path_game_file, LIST_CHOSEN_EVENT)

                # Process more feature
                game_df['shot_distance'] = game_df.apply(compute_shot_distance, axis=1)
                game_df['shot_angle'] = game_df.apply(compute_shot_angle, axis=1)
                game_df['isgoal'] = game_df['event type'].apply(lambda x: 1 if x == 'goal' else 0)
                game_df['is empty net'] = game_df.apply(check_empty_net, axis=1)

                # Concat dateframe of current game (game_df) into final dataframe (df)
                df = pd.concat([df, game_df], ignore_index=True)

        print(f"[INFO] Done process season {season}")

    df = df.dropna()
    print(f"Shape of output df: {df.shape}")
    df.to_csv(PATH_OUTPUT_FILE)