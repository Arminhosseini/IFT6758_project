import sys
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from game_client import *
from serving_client import ServingClient


def visualize_shot_map(game_df, home_team_name, away_team_name):
    """
    This function will visualize the shot map to the front end
    * Arguments:
    game_df -- a dataframe, indicate the information of the game
    
    * Returns: None
    """

    st.header('Shot map')
    home_shots = game_df[game_df['team'] == 'home']
    away_shots = game_df[game_df['team'] == 'away']

    fig, ax = plt.subplots(figsize=(15, 7))
    x_limits = (-100, 100)
    y_limits = (-42.5, 42.5)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    ax.scatter(home_shots['x coor'], home_shots['y coor'], c='red', label=f'{home_team_name} Shots', edgecolors='w')
    ax.scatter(away_shots['x coor'], away_shots['y coor'], c='blue', label=f'{away_team_name} Shots', edgecolors='w')
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.axhline(y=0, color='grey', linestyle='--')

    ax.legend()
    ax.set_title(f'Shot Map for {home_team_name} and {away_team_name}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    st.pyplot(fig)
    st.write(
        """
        The shot map above visualizes the location of shots taken by both the home and away teams during the game. 
        Each dot represents a shot, with red dots indicating shots by the home team and blue dots for the away team. 
        The position of each dot on the map corresponds to the shot's location on the rink.
        """
    )


def get_features(model_name):
    """
    This function get the features by inspect the model name

    * Arguments:
    model_name -- a string.
    
    * Return:
    features -- a list, indicate a list of featues
    """

    if model_name == 'log_reg_dist':
        features = ["shot_distance"]
        st.sidebar.success(f"Success load model {model_name}")
    elif model_name == 'log_reg_ang':
        features = ["shot_angle"]
        st.sidebar.success(f"Success load model {model_name}")
    elif model_name == 'log_reg_dist_ang':
        features = ["shot_distance", "shot_angle"]
        st.sidebar.success(f"Success load model {model_name}")
    else:
        st.sidebar.error("Please choose model")
        features = None

    return features


SERVING_IP = os.environ.get("SERVING_IP", "serving")
# SERVING_PORT = int(os.environ.get("SERVING_POST", '5000'))
SERVING_PORT = 5000
# BASE_URL = f"http://{SERVING_IP}:{SERVING_PORT}"

# Folder to store game data 
PATH_OUTPUT_FOLDER = os.path.join("game_client_data")
if os.path.exists(PATH_OUTPUT_FOLDER) == False:
    os.mkdir(PATH_OUTPUT_FOLDER)    

game_df_pred = pd.DataFrame()
features = ["shot_distance"]

serving_client = ServingClient(ip=SERVING_IP, port=SERVING_PORT, features=features)


if __name__ == "__main__":

    # ===================================== Side bar frontend ======================================
    st.sidebar.header('Workspace')
    workspace = st.sidebar.selectbox("Select Workspace", options=['ift6758-b09-project'])

    st.sidebar.header('Model')
    model_name = st.sidebar.selectbox("Select Model", options=['log_reg_dist', 'log_reg_ang', 'log_reg_dist_ang'])

    st.sidebar.header('Version')
    version = st.sidebar.selectbox("Select Version", options=["1.0.0"])

    get_model_button = st.sidebar.button('Get model')

    # Get features name and update serving client
    if get_model_button:
        features = get_features(model_name)
        download_info = serving_client.download_registry_model(workspace, model_name, version)
        serving_client.features = features


    # ====================================== Main area ======================================
    st.title('Hockey Visualization App')

    game_id = st.text_input("Enter Game ID", "2023020510")
    # game_id = st.text_input("Enter Game ID")

    if st.button('Ping game'):
        features = get_features(model_name)
        serving_client.features = features

        file_predict_name = f"{game_id}_{'_'.join(features)}.csv"
        path_output_file_predict = os.path.join(PATH_OUTPUT_FOLDER, file_predict_name)

        # Check if we already predict it
        if os.path.exists(path_output_file_predict):  
            game_df_pred = pd.read_csv(path_output_file_predict)   # Read previous game 

            # Get data of new game
            try: json_game_data = get_game_data_json(game_id)
            except Exception as error:
                st.error(f"The game id {game_id} is NOT exist.")
                sys.exit()
                
            (home_team_name, away_team_name) = get_team_name(json_game_data)

            # Process feature of game data
            game_df = process_feature(json_game_data)  

            game_df_pred = handle_saved_game_id(game_df_pred, game_df, serving_client)

            game_df_pred.to_csv(path_output_file_predict)

        else:
            # Get data of game
            try: json_game_data = get_game_data_json(game_id)
            except Exception as error:
                st.error(f"The game id {game_id} is NOT exist.")
                sys.exit()

            (home_team_name, away_team_name) = get_team_name(json_game_data)
            game_df = process_feature(json_game_data)

            list_output = serving_client.predict(game_df)

            game_df_pred = pd.concat([game_df, list_output], axis=1)
            game_df_pred.to_csv(path_output_file_predict)

        # Get predicted and actual goal
        goals = get_actual_goal(json_game_data)
        home_actual_goal = goals['homeScore']
        away_actual_goal = goals['awayScore']
        home_expected_goal = round(game_df_pred[game_df_pred['team'] == 'home']['y_pred'].sum(), 2)
        away_expected_goal = round(game_df_pred[game_df_pred['team'] == 'away']['y_pred'].sum(), 2)

        # Get info of last event 
        try:
            last_event = get_last_event(json_game_data)
            if last_event['typeDescKey'] == "game-end":
                st.text(f"Period {last_event['period']} - {last_event['timeRemaining']} left (GAME END)")
            else:
                st.text(f"Period {last_event['period']} - {last_event['timeRemaining']} left")
        except:
            pass

        col1, col2 = st.columns(2)
        with col1:
            st.header(f"{home_team_name}")
            st.markdown("<span style='color:red; font-size: smaller;'>Home Team</span>", unsafe_allow_html=True)
            home_delta = round(home_actual_goal - home_expected_goal, 2)
            st.metric(label='xG (actual)', value=f'{home_expected_goal} ({home_actual_goal})', delta=home_delta)
        with col2:
            st.header(f"{away_team_name}")
            st.markdown("<span style='color:blue; font-size: smaller;'>Away Team</span>", unsafe_allow_html=True)
            away_delta = round(away_actual_goal - away_expected_goal, 2)
            st.metric(label='xG (actual)', value=f'{away_expected_goal} ({away_actual_goal})', delta=away_delta)

        # Show prediction to screen
        st.header('Data used for predictions (and predictions)')
        # if "Unnamed: 0" in game_df_pred.columns: game_df_pred.drop(columns=['Unnamed: 0'], inplace=True)
        game_df_pred = game_df_pred.loc[:, ~game_df_pred.columns.str.contains('^Unnamed')]
        st.dataframe(game_df_pred)

        # Visualize shot map
        visualize_shot_map(game_df_pred, home_team_name, away_team_name)
    else:
        st.error(f"Please enter game id")

