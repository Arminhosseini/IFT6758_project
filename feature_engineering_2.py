import os
from dotenv import load_dotenv
load_dotenv()
import comet_ml
from comet_ml import Artifact
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import json
from datetime import datetime
from crawler import Crawler
class Feature_Engineering_2:
    def __init__(self):
        pass

    def Calculate_Distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        try:
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        except:
            distance = None
        return distance

    def Time_String_To_Seconds(self, time_str):
        """
        This function convert from mm:ss to the total number of seconds. For example: "03:45" -> 225
        """
        try:
            minutes, seconds = map(int, time_str.split(':'))
        except:
            print(f"Error in time string. Current time string: {time_str}")
        total_seconds = minutes * 60 + seconds
        return total_seconds

    def Retrieve_File_Name_From_gamePk(self, gamePk, parent_directory="Dataset"):
        season = str(gamePk[0:4])
        type_game = str(gamePk[4:6])
        if type_game == "02":
            type_game = "regular_season"
        elif type_game == "03":
            type_game = "playoffs"
        path_json_file = os.path.join(parent_directory, season, type_game, f"{gamePk}.json")
        return path_json_file


    def Get_List_Event_From_Json(self, path_json_file):
        with open(path_json_file, 'r') as json_file:
            data = json.load(json_file)
        try:
            list_event = data['liveData']['plays']['allPlays']
        except:
            list_event = None

        return list_event


    def Extract_Previous_Event(self, list_event, shot_sample_period, shot_sample_periodTime):
        previous_event = None
        for (idx_event, event) in enumerate(list_event):
            event_period= str(event['about']['period']).lower()
            event_periodTime = str(event['about']['periodTime']).lower()
            event_name = str(event['result']['event']).lower()
            if (event_period == shot_sample_period) and (event_periodTime == shot_sample_periodTime) and ((event_name == "shot") or (event_name == "goal")):
                previous_event = list_event[idx_event - 1]
                break
            else:
                continue
        return previous_event


    def Get_Info_Previous_Event(self, previous_event, default_value=None):
        event_type = previous_event['result']['event']
        try:
            coor_x = previous_event['coordinates']['x']
            coor_y = previous_event['coordinates']['y']
        except:
            coor_x = coor_y = default_value
        event_period = previous_event['about']['period']
        event_periodTime = previous_event['about']['periodTime']

        return (event_type, coor_x, coor_y, event_period, event_periodTime)


    def Calculate_Time_Distance(self, time_str1, time_str2):
        time_format = "%M:%S"
        time1 = datetime.strptime(time_str1, time_format)
        time2 = datetime.strptime(time_str2, time_format)

        time_difference_seconds = abs((time1 - time2).total_seconds())
        return time_difference_seconds


    def Add_InFo_Previous_Event(self, df):
        list_last_event_type = []
        list_coor_x_last_event = []
        list_coor_y_last_event = []
        list_time_last_event = []
        list_distance_last_event = []

        cache_gamePk = None
        cache_list_event = None

        for idx_sample, shot_sample in df.iterrows():
            # if idx_sample % 10_000 == 0:
            #     print(f"[INFO] Idx = {idx_sample}")

            try:
                shot_sample_gamePk = str(shot_sample['gamePk']).lower()
                shot_sample_period = str(shot_sample['period']).lower()
                shot_sample_periodTime = str(shot_sample['periodTime']).lower()
                current_coor_x = shot_sample['x-coordinate']
                current_coor_y = shot_sample['y-coordinate']
                current_period = shot_sample['period']
                current_periodTime = shot_sample['periodTime']

                if shot_sample_gamePk == cache_gamePk:
                    list_event = cache_list_event
                else:
                    path_json_file = self.Retrieve_File_Name_From_gamePk(shot_sample_gamePk)
                    list_event = self.Get_List_Event_From_Json(path_json_file)

                previous_event = self.Extract_Previous_Event(list_event, shot_sample_period, shot_sample_periodTime)
                (previous_event_type, previous_coor_x, previous_coor_y, previous_event_period, previous_event_periodTime) = self.Get_Info_Previous_Event(previous_event)
                if previous_coor_x == None:
                    previous_coor_x = current_coor_x
                if previous_coor_y == None:
                    previous_coor_y = current_coor_y

                list_last_event_type.append(previous_event_type)
                list_coor_x_last_event.append(previous_coor_x)
                list_coor_y_last_event.append(previous_coor_y)

                # Time distance 
                if int(current_period) == int(previous_event_period):
                    time_distance = self.Calculate_Time_Distance(current_periodTime, previous_event_periodTime)
                    list_time_last_event.append(time_distance)
                elif int(current_period) > int(previous_event_period):
                    time_distance = self.Calculate_Time_Distance(previous_event_periodTime, "20:00") + self.Calculate_Time_Distance("00:00", current_periodTime)
                    list_time_last_event.append(time_distance)
                else:
                    print(f"Error at index {idx_sample}")
                    print(f"Current time: {current_periodTime}, Previous time: {previous_event_periodTime}")

                # Distance from the last event to current event
                distance = self.Calculate_Distance((current_coor_x, current_coor_y), (previous_coor_x, previous_coor_y))
                list_distance_last_event.append(distance)

                # Cache for next calculation
                cache_list_event = list_event
                cache_gamePk = shot_sample_gamePk

            except Exception as error:
                print(f"Error at index {idx_sample}")
                print(f"Error: {error}")
                break



        assert len(list_last_event_type) == len(list_coor_x_last_event) == \
                len(list_coor_y_last_event) == len(list_time_last_event) == len(list_distance_last_event)

        df['last_event_type'] = list_last_event_type
        df['coor_x_last_event'] = list_coor_x_last_event
        df['coor_y_last_event'] = list_coor_y_last_event
        df['time_last_event'] = list_time_last_event
        df['distance_last_event'] = list_distance_last_event

        return df


class Feature_Engineering_3:
    def __init__(self):
        pass

    def add_feature_3(self, df):
        # Create new columns 'is_rebound', 'Change in shot angle', and 'Speed'
        df['is_rebound'] = False
        df['Change in shot angle'] = 0.0
        df['Speed'] = 0.0

        # Apply condition logic to each row
        for index, row in df.iterrows():
            # Check if the previous event is 'Shot' and if the game period is the same as the previous period
            if index > 0 and df.loc[index - 1, 'eventType'] == 'Shot' and \
                    df.loc[index, 'period'] == df.loc[index - 1, 'period']:
                df.at[index, 'is_rebound'] = True
                # Calculate the change in shot angle
                prev_angle = df.loc[index - 1, 'angle']
                current_angle = df.loc[index, 'angle']
                df.at[index, 'Change in shot angle'] = current_angle - prev_angle

            # Calculate the speed
            if index > 0:
                distance_from_last_event = df.loc[index, 'distance_last_event']
                time_from_last_event = df.loc[index, 'time_last_event']
                if time_from_last_event != 0:
                    df.at[index, 'Speed'] = distance_from_last_event / time_from_last_event
        return df


# Class definition for Penalty
class Penalty:
    def __init__(self, penalty_min, game_time):
        # A dictionary mapping penalty durations to their corresponding types
        min_to_type = {"2": "minor", "4": "double", "5": "major"}
        # Initialize the Penalty object with penalty duration and game time
        self.penalty_min = penalty_min
        # Determine the penalty type based on the duration
        self.type = min_to_type[str(penalty_min)]
        # Calculate the end time of the penalty by adding the penalty duration
        self.end_time = game_time + penalty_min * 60


# Class definition for Team
class Team:
    def __init__(self, name):
        # Initialize Team object with a name
        self.name = name

        # Set the initial number of players to 5
        self.n_players = 5

        # Initialize lists to store current penalties and reserved penalties
        self.penalties = list()
        self.reserved_penalties = list()

    # Assign penalty to team
    def add_penalty(self, penalty: Penalty):
        # Check if there are more than 3 players on the team
        if self.n_players > 3:
            # Add the penalty to the current penalties list
            self.penalties.append(penalty)
            # Decrement the number of players
            self.n_players -= 1
        else:
            # If the team has 3 or fewer players, add the penalty to the reserved penalties list
            self.reserved_penalties.append(penalty)

    # Remove the penalty from the active penalties list
    def remove_penalty(self, penalty):
        self.penalties.remove(penalty)
        # Increase the number of players in the team
        self.n_players += 1

        # Check if there are reserved penalties
        if len(self.reserved_penalties) > 0:
            # Pop the last reserved penalty
            new_p = self.reserved_penalties.pop()
            # Add a new penalty to the active penalties list based on the popped reserved penalty
            self.penalties.append(Penalty(new_p.penalty_min, penalty.end_time))
            # Decrease the number of players in the team
            self.n_players -= 1

        # Return the end time of the removed penalty
        return penalty.end_time


class Feature_Engineering_4:
    def __init__(self):
        # Create an instance of the Crawler class
        self.crawler = Crawler()

    # Function to remove penalties that have expired
    def remove_penalty(self, team: Team, time):
        latest_removal = 0
        for p in team.penalties:
            if p.end_time <= time:
                end_time = team.remove_penalty(p)
                if end_time > latest_removal:
                    latest_removal = end_time
        return latest_removal

    # Function to set the start time of a power play
    def set_power_play(self, new_start, home: Team, away: Team, prev_start):
        if home.n_players != away.n_players and prev_start == 0:
            return new_start
        if home.n_players == away.n_players:
            return 0
        return prev_start

    # Function to extract bonus features from the game data
    def get_bonus_features(self, game_id):
        raw_data = self.crawler.read_data_by_game_id(game_id)

        # Extract relevant information from the raw data
        game_id = raw_data["gamePk"]
        home_name = raw_data["gameData"]["teams"]["home"]["name"]
        away_name = raw_data["gameData"]["teams"]["away"]["name"]

        # Initialize Team objects for the home and away teams
        home = Team(home_name)
        away = Team(away_name)

        # Initialize the start time of the power play
        power_play_start = 0
        # List to store bonus features for each play
        plays_bonus_features = list()
        # Iterate through each play in the game
        for play in raw_data["liveData"]["plays"]["allPlays"]:
            # Extract information about the play
            period = play["about"]["period"]
            period_time = play["about"]["periodTime"]
            time = (period - 1) * 20 * 60
            time += sum(
                int(x) * 60 ** i for i, x in enumerate(period_time.split(":")[::-1])
            )

            # Remove expired penalties from the home teams
            latest_removal = self.remove_penalty(home, time)
            power_play_start = self.set_power_play(latest_removal, home, away, power_play_start)

            # Remove expired penalties from the away teams
            latest_removal = self.remove_penalty(away, time)
            power_play_start = self.set_power_play(latest_removal, home, away, power_play_start)

            # Process plays based on their event type
            if play["result"]["event"] == "Penalty":
                # Determine the penalty team and its minutes
                team = home if play["team"]["name"] == home_name else away
                penalty_min = play["result"]["penaltyMinutes"]

                # Add penalty to the team's penalties if it's not a 10-minute or 0-minute penalty
                if penalty_min != 10 and penalty_min != 0:
                    team.add_penalty(Penalty(penalty_min, time))
                    # Update power play start time if the teams have different player counts
                    if home.n_players != away.n_players:
                        power_play_start = time

            event_type = play["result"]["event"]
            if event_type == "Goal":
                team = home if play["team"]["name"] == home_name else away
                other_team = home if team == away else away

                # Calculate power play time and adjust based on player counts
                power_play_time = time - power_play_start if power_play_start != 0 else 0
                n_friend = team.n_players
                n_oppose = other_team.n_players

                if n_friend < n_oppose:
                    power_play_time *= -1

                # Append bonus features for a goal to the plays_bonus_features list
                plays_bonus_features.append(
                    (period, period_time, event_type, power_play_time, n_friend, n_oppose)
                )

                # Remove a minor penalty from the opposing team (if applicable)
                if team.n_players != other_team.n_players:
                    for p in other_team.penalties:
                        if p.type == "minor":
                            latest_removal = other_team.remove_penalty(p)
                            power_play_start = self.set_power_play(
                                latest_removal, home, away, power_play_start
                            )
                            break
                        # Convert double minor to a regular minor if more than 2 minutes remain
                        elif p.type == "double":
                            if p.end_time - time > 120:
                                p = Penalty(2, time)
                            else:
                                latest_removal = other_team.remove_penalty(p)
                                power_play_start = self.set_power_play(
                                    latest_removal, home, away, power_play_start
                                )
                            break

            if event_type == "Shot":
                team = home if play["team"]["name"] == home_name else away
                other_team = home if team == away else away

                # Calculate power play time and adjust based on player counts
                power_play_time = time - power_play_start if power_play_start != 0 else 0
                n_friend = team.n_players
                n_oppose = other_team.n_players

                if n_friend < n_oppose:
                    power_play_time *= -1

                # Append bonus features for a shot to the plays_bonus_features list
                plays_bonus_features.append(
                    (period, period_time, event_type, power_play_time, n_friend, n_oppose)
                )

        # Return the list of bonus features for each play in the game
        return plays_bonus_features

    def add_bonus_feats(self, tidy_data):
        tidy_data = tidy_data.drop("Unnamed: 0", axis=1, errors="ignore")
        tidy_data[["power_play_time", "n_friend", "n_oppose"]] = None, None, None

        # Get unique game IDs from the tidy data
        game_ids = tidy_data["gamePk"].unique()
        i = 0
        for game_id in game_ids:
            # Get shots and goals events for each game_id
            plays = tidy_data[tidy_data["gamePk"] == game_id]
            plays_bonus_features = self.get_bonus_features(game_id)
            # An error happens if number of mentioned events are not equal in
            # previous tidy data and plays_bonus_features
            if len(plays) != len(plays_bonus_features):
                print("Error in game_id:", game_id)
                print("Number of plays:", len(plays))
                print("Number of bonus_features:", len(plays_bonus_features))
                break
            # Add new features to tidy data
            for bonus_feature in plays_bonus_features:
                tidy_data.at[i, "power_play_time"] = bonus_feature[3]
                tidy_data.at[i, "n_friend"] = bonus_feature[4]
                tidy_data.at[i, "n_oppose"] = bonus_feature[5]
                i += 1
        return tidy_data


def split_train_test(df):
    df['year'] = df['gamePk'].astype(str).str[:4].astype(int)
    df['gameType'] = df['gamePk'].astype(str).str[4:6]
    train = df[(df['year'].isin([2016, 2017, 2018, 2019])) & (df['gameType'] == '02')]
    test_regular = df[(df['year'].isin([2020])) & (df['gameType'] == '02')]
    test_playoffs = df[(df['year'].isin([2020])) & (df['gameType'] == '03')]

    return train, test_regular, test_playoffs

def Time_String_To_Seconds(time_str):
    """
    This function convert from mm:ss to the total number of seconds. For example: "03:45" -> 225
    """
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds

def remove_extra_features(df):
    df = df[['periodTime', 'period', 'x-coordinate', 'y-coordinate', 'attackingSide', 'shot_distance', 'angle',
             'isEmptyNet', 'shotType', 'last_event_type', 'coor_x_last_event', 'coor_y_last_event', 'time_last_event',
             'distance_last_event', 'is_rebound', 'Change in shot angle', 'Speed', 'power_play_time', 'n_friend',
             'n_oppose', 'isgoal']]
    df['period_second'] = df['periodTime'].apply(Time_String_To_Seconds)
    df["game_second"] = (df['period'] - 1) * 60 * 20 + df['period_second']

    return df
def main():

    PATH_TIDY_DATA_CSV = r"Dataset/feature_engineering1.csv"

    df = pd.read_csv(PATH_TIDY_DATA_CSV)
    print(f"[INFO] Read df from: {PATH_TIDY_DATA_CSV}")

    fe2 = Feature_Engineering_2()
    df_fe2 = fe2.Add_InFo_Previous_Event(df)
    print("[INFO] Feature Engineering 2.1 is completed!")

    fe3 = Feature_Engineering_3()
    df_fe3 = fe3.add_feature_3(df_fe2)
    print("[INFO] Feature Engineering 2.2 is completed!")

    fe4 = Feature_Engineering_4()
    df_fe4 = fe4.add_bonus_feats(df_fe3)
    print("[INFO] Feature Engineering (Bonus) is completed!")

    path_to_csv = os.path.join("Dataset", "feature_engineering2.csv")
    df_fe4.to_csv(path_to_csv)

    train, test_regular, test_playoffs = split_train_test(df_fe4)
    train = remove_extra_features(train)
    test_regular = remove_extra_features(test_regular)
    test_playoffs = remove_extra_features(test_playoffs)

    path_train_csv = os.path.join("Dataset", "train.csv")
    path_test_regular_csv = os.path.join("Dataset", "test_regular.csv")
    path_test_playoff_csv = os.path.join("Dataset", "test_playoff.csv")

    train.to_csv(path_train_csv)
    test_regular.to_csv(path_test_regular_csv)
    test_playoffs.to_csv(path_test_playoff_csv)
    print(f"[INFO] All the data are saved!")

    workspace_name = r"ift6758-b09-project"
    project_name = r"ift6758-project-milestone2"
    slzhou_api_key = os.environ.get('COMET_API_KEY')

    experiment = comet_ml.Experiment(api_key=slzhou_api_key, project_name=project_name, workspace=workspace_name)

    artifact = Artifact(name="dataset", artifact_type="dataset", version="4.0.2")
    artifact.add(path_train_csv)
    artifact.add(path_test_regular_csv)
    artifact.add(path_test_playoff_csv)

    experiment.log_artifact(artifact)

    experiment.end()
    print(f"Done store feature engineering 2 dataframe")


main()
