import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# def Convert_Coord_NHL_2_Image(list_x_coor, list_y_coor, image_width = 200, image_height = 85, nhl_rink_width = 200, nhl_rink_length = 85):
#     """Function convert from NHL coordinate ice rink to Image coordiante

#     Args:
#         nhl_x (integer): coordinate of the shot in NHL rink 
#         nhl_y (integer): coordinate of the shot in NHL rink 

#     Returns:
#         (image_x, image_y) (int, int): the coordinate of in Image space
#     """
    
#     transformation_matrix = np.array([
#         [1, 0, image_width / 2],
#         [0, -1, image_height / 2],  
#         [0, 0, 1]
#     ])
    
#     for (i, (nhl_x_coor, nhl_y_coor)) in enumerate(zip(list_x_coor, list_y_coor)):
    
#         # Convert NHL coordinates to image coordinates
#         image_x = (nhl_x_coor / nhl_rink_width) * image_width
#         image_y = (nhl_y_coor / nhl_rink_length) * image_height 
        
#         image_matrix_coor = np.dot(transformation_matrix, np.array([image_x, image_y, 1]))
#         image_x = image_matrix_coor[0]
#         image_y = image_matrix_coor[1]
        
#         image_x -= image_width/2
        
#         list_x_coor[i] = int(image_x)
#         list_y_coor[i] = int(image_y)
        
#     return (list_x_coor, list_y_coor)


def Correct_Side_Rink_Coordinate(list_x_coor, list_y_coor):
    """This function correct side of the rink when change period

    Args:
        list_x_coor (list or numpy array): 
        list_y_coor (_type_): _description_

    Returns:
        (list_x_coor, list_y_coor): 
    """
    for (i, (x_coor, y_coor)) in enumerate(zip(list_x_coor, list_y_coor)):
        if x_coor < 0:
            list_x_coor[i] = -x_coor
            list_y_coor[i] = -y_coor
    
    return (list_x_coor, list_y_coor)


if __name__ == "__main__":
    
    # =========================== Hyper Parameter ===========================
    path_tidy_data_csv = os.path.join("Dataset", "tidyData.csv")
    # path_tidy_data_csv = r"Dataset\tidyData.csv"
    
    bin_width = 5
    
    # image_width = 200
    # image_height = 85
    
    nhl_rink_width = 200
    nhl_rink_height = 85
    
    # extent_coordinate = [0, nhl_rink_width//2, -nhl_rink_height//2, nhl_rink_height//2]
    extent_coordinate = [-nhl_rink_height//2, nhl_rink_height//2, 0, nhl_rink_width//2]
    
    game_types = {
            "preseason": "01",
            "regular_season": "02",
            "playoffs": "03",
            "all-star": "04",
    }
    
    ice_rink_image_path = os.path.join("images", "advanced_visualization", "half_nhl_rink.png")
    
    alpha = 0.3
    
    path_folder_shot_map = os.path.join("images", "advanced_visualization")

    # ========================================================================
   
    # 1. Read df
    df = pd.read_csv(path_tidy_data_csv, low_memory=False)

    # 2. Read shot df
    # df_shot = df[df['eventType'] == "Shot"]
    df_shot = df
    df_shot['gamePk'] = df_shot['gamePk'].astype(str)
    df_shot = df_shot.dropna(subset=['x-coordinate', 'y-coordinate']) # Drop row with NaN value
    df_shot.reset_index(drop=True, inplace=True)

    df_shot['x-coordinate'] = pd.to_numeric(df_shot['x-coordinate'])
    df_shot['y-coordinate'] = pd.to_numeric(df_shot['y-coordinate'])

    
    # 3. Loop through all season
    list_season = ["2016", "2017", "2018", "2019", "2020"]
    list_team_names = ['Montréal Canadiens', 'New York Rangers', 'Ottawa Senators', 'Boston Bruins',\
                        'Washington Capitals', 'Toronto Maple Leafs', 'Pittsburgh Penguins', 'Columbus Blue Jackets',\
                        'Chicago Blackhawks', 'Nashville Predators', 'Minnesota Wild', 'St. Louis Blues', 'Anaheim Ducks',\
                        'Calgary Flames', 'San Jose Sharks', 'Edmonton Oilers', 'Los Angeles Kings', 'Buffalo Sabres',\
                        'New York Islanders', 'Detroit Red Wings', 'Tampa Bay Lightning', 'New Jersey Devils', 'Florida Panthers',\
                        'Carolina Hurricanes', 'Winnipeg Jets', 'Dallas Stars', 'Philadelphia Flyers', 'Colorado Avalanche',\
                        'Arizona Coyotes', 'Vancouver Canucks', 'Vegas Golden Knights']
    
    # list_season = ["2017"]
    # list_team_names = ['San Jose Sharks']
    
    for season in list_season:
        path_folder_shot_map_season = os.path.join(path_folder_shot_map, season)
        if os.path.exists(path_folder_shot_map_season) == False:
            os.mkdir(path_folder_shot_map_season)
    
        # 3.0 Get current season df
        target_search_string = f"{season}"
        current_season_df = df_shot[df_shot['gamePk'].str.startswith(target_search_string)]
        current_season_df.reset_index(drop=True, inplace=True)

        # 3.1 Get coordinate of current season
        list_x_coor_season = np.array(current_season_df['x-coordinate'])
        list_y_coor_season = np.array(current_season_df['y-coordinate'])

        # 3.2 Process coordinate of current season
        (list_x_coor_season, list_y_coor_season) = Correct_Side_Rink_Coordinate(list_x_coor_season, list_y_coor_season)
        # (list_x_coor_season, list_y_coor_season) = Convert_Coord_NHL_2_Image(list_x_coor_season, list_y_coor_season)
        
        # 3.3 Create a 2D histogram with 10 x 10 bins
        # bins = [np.arange(0, image_width//2 + 1, bin_width), np.arange(0, image_height + 1, bin_width)]
        bins = [np.arange(0, nhl_rink_width//2 + bin_width, bin_width),\
                np.arange(-nhl_rink_height//2, nhl_rink_height//2 + bin_width, bin_width)]
        hist_season, x_edges, y_edges = np.histogram2d(list_x_coor_season, list_y_coor_season, bins=bins)

        # 3.4 Calculate league shot rate per hour
        total_num_game_season = len(current_season_df['gamePk'].unique())
        hist_season = hist_season / (total_num_game_season*2)
        # hist_season = gaussian_filter(hist_season, sigma=3) # Smooth histogram
        
        # 3.5 Loop through all team
        for team_name in list_team_names:
        
            # --- a. Extract df of this team in current season
            df_team = current_season_df[current_season_df['teamName'] == team_name]
            if len(df_team) == 0:
                continue
            
            # --- b. Get (x, y) coordinate of shot in this team
            list_x_coor_team = np.array(df_team['x-coordinate'])
            list_y_coor_team = np.array(df_team['y-coordinate'])
            
            # --- c. Process coordinate 
            (list_x_coor_team, list_y_coor_team) = Correct_Side_Rink_Coordinate(list_x_coor_team, list_y_coor_team)
            # (list_x_coor_team, list_y_coor_team) = Convert_Coord_NHL_2_Image(list_x_coor_team, list_y_coor_team)
            
            # --- d. Create a 2D histogram with 10 x 10 bins
            hist_team, x_edges, y_edges = np.histogram2d(list_x_coor_team, list_y_coor_team, bins=bins)

            # --- e. Calculate shot rate per hour of this team
            total_num_game = len(df_team['gamePk'].unique())
            hist_team = hist_team / total_num_game

            # --- f. Compare this team vs league about shot rate per hour
            hist_diff = hist_team - hist_season
            hist_diff = gaussian_filter(hist_diff, sigma=1.5) # Smooth histogram

            # --- g. Plot 
            plt.clf()
            
            image = Image.open(ice_rink_image_path)
            image = np.rot90(image)
            plt.imshow(image, alpha=alpha, extent=extent_coordinate) # Plot ice rink

            plt.xlabel('Distance from the center red line (ft)')
            plt.ylabel('Distance from the center of rink (ft)')
            plt.title(f"{team_name} Offence \n Shot rate per hour compared to league \n Season {season}-{int(season)+1}")
            
            data_min= hist_diff.min()
            data_max= hist_diff.max()

            if abs(data_min) > data_max:
                data_max = data_min * -1
            elif data_max > abs(data_min):
                data_min = data_max * -1

            hist_diff = np.rot90(hist_diff) # Rotate
            plt.contourf(hist_diff.T, extent=extent_coordinate, vmin=data_min, vmax=data_max,\
                        cmap='RdBu_r', levels = np.linspace(data_min, data_max, 12), alpha=1-alpha)
            plt.colorbar(label='Excess shot rate per hours')
            
            # --- h. Save shot map image
            file_name_shot_map = f"{team_name}.jpg"
            path_file_shot_map = os.path.join(path_folder_shot_map_season, file_name_shot_map)
            plt.savefig(path_file_shot_map)
            plt.close()
    
        print(f"Done season {season}")
    print(f"[INFO] The shot maps are stored in the foldler images/advanced_visualization")