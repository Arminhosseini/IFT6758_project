import streamlit as st
import requests
from dotenv import load_dotenv
from game_client import *
from serving_client import ServingClient


SERVING_IP = os.environ.get("SERVING_IP", "serving")
SERVING_PORT = 5000
# SERVING_PORT = os.environ.get("SERVING_PORT", "5000")
# BASE_URL = f"http://{SERVING_IP}:{SERVING_PORT}"

PATH_OUTPUT_FOLDER = os.path.join("game_client_data")
game_df_pred = pd.DataFrame()
features = ["shot_distance"]


if __name__ == "__main__":

    # ===================================== Side bar frontend ======================================
    st.sidebar.header('Workspace')
    workspace = st.sidebar.selectbox("Select Workspace", options=['ift6758-b09-project'])

    st.sidebar.header('Model')
    model_name = st.sidebar.selectbox("Select Model", options=['log_red_dist', 'log_reg_ange', 'log_reg_dist_ang'])

    st.sidebar.header('Version')
    version = st.sidebar.selectbox("Select Version", options=["1.0.0"])

    if st.sidebar.button('Get model'):
        if model_name == 'log_red_dist':
            features = ["shot_distance"]
            st.sidebar.success(f"Success load model {model_name}")
        elif model_name == 'log_reg_ange':
            features = ["shot_angle"]
            st.sidebar.success(f"Success load model {model_name}")
        elif model_name == 'log_reg_dist_ang':
            features = ["shot_distance", "shot_angle"]
            st.sidebar.success(f"Success load model {model_name}")
        else:
            st.sidebar.error("Please choose model")
    # =============================================================================================


    # Load model 
    serving_client = ServingClient(ip=SERVING_IP, port=SERVING_PORT, features=features)
    download_info = serving_client.download_registry_model(workspace, model_name, version)


    # ====================================== Main area ======================================
    st.title('Hockey Visualization App')

    game_id = st.text_input("Enter Game ID", "2022030411")
    # game_id = st.text_input("Enter Game ID")

    if st.button('Ping game'):
        file_predict_name = f"{game_id}_{''.join(features)}.csv"
        path_output_file_predict = os.path.join(PATH_OUTPUT_FOLDER, file_predict_name)

        # Check if we already predict it
        if os.path.exists(path_output_file_predict):  
            game_df_pred = pd.read_csv(path_output_file_predict)
            path_out_json_game_data = get_game_data_and_save_2_json(game_id, PATH_OUTPUT_FOLDER)
            (home_team_name, away_team_name) = get_team_name(path_out_json_game_data)

        else:
            path_out_json_game_data = get_game_data_and_save_2_json(game_id, PATH_OUTPUT_FOLDER)
            (home_team_name, away_team_name) = get_team_name(path_out_json_game_data)
            game_df = process_feature(path_out_json_game_data)

            # serving_client = ServingClient(features=features)
            list_output = serving_client.predict(game_df)

            game_df_pred = pd.concat([game_df, list_output], axis=1)
            game_df_pred.to_csv(path_output_file_predict)


        goals = get_actual_goal(path_out_json_game_data)

        home_actual_goal = goals['homeScore']
        away_actual_goal = goals['awayScore']

        home_expected_goal = game_df_pred[game_df_pred['team'] == 'home']['y_pred'].sum()
        home_expected_goal = round(home_expected_goal, 2)
        away_expected_goal = game_df_pred[game_df_pred['team'] == 'away']['y_pred'].sum()
        away_expected_goal = round(away_expected_goal, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.header(f"{home_team_name}")
            home_delta = home_actual_goal - home_expected_goal
            st.metric(label='xG (actual)', value=f'{home_expected_goal} ({home_actual_goal})', delta=home_delta)

        with col2:
            st.header(f"{away_team_name}")
            away_delta = away_actual_goal - away_expected_goal
            st.metric(label='xG (actual)', value=f'{away_expected_goal} ({away_actual_goal})', delta=away_delta)

        # Section for data used for predictions
        st.header('Data used for predictions (and predictions)')
        if "Unnamed: 0" in game_df_pred.columns: game_df_pred.drop(columns=['Unnamed: 0'], inplace=True)
        st.dataframe(game_df_pred)

    else:
        st.error(f"Please enter game id")

