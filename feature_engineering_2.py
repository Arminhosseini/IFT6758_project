import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from PIL import Image
import matplotlib.pyplot as plt
import math
import json
from datetime import datetime, timedelta


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
            if idx_sample % 10_000 == 0:
                print(f"[INFO] Idx = {idx_sample}")

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
        
        print(len(list_last_event_type))
        print(len(list_coor_x_last_event))
        print(len(list_coor_y_last_event))
        print(len(list_time_last_event))
        print(len(list_distance_last_event))

        assert len(list_last_event_type) == len(list_coor_x_last_event) == \
                len(list_coor_y_last_event) == len(list_time_last_event) == len(list_distance_last_event)

        df['last_event_type'] = list_last_event_type
        df['coor_x_last_event'] = list_coor_x_last_event
        df['coor_y_last_event'] = list_coor_y_last_event
        df['time_last_event'] = list_time_last_event
        df['distance_last_event'] = list_distance_last_event

        return df


def main():

    PATH_TIDY_DATA_CSV = r"Dataset/feature_engineering1.csv"

    df = pd.read_csv(PATH_TIDY_DATA_CSV)
    print(f"Shape of df: {df.shape}")

    fe2 = Feature_Engineering_2()
    df_fe2 = fe2.Add_InFo_Previous_Event(df)

    path_output_csv = os.path.join("Dataset", "tidyData_fe2.csv")
    df_fe2.to_csv(path_output_csv)


main()