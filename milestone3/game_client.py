import json
import requests
import pandas as pd
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from new_crawler import Crawler


if __name__ == "__main__":

    path_output_folder = os.path.join("game_client_data")
    game_id = r"2017030111"

    # Get game data
    c = Crawler() 
    json_game_data = c.get_game_data(game_id=game_id)

    # Store game data to json file
    path_out_json_game_data = os.path.join(path_output_folder, f"{game_id}.json")
    with open(path_out_json_game_data, "w") as file:
        json.dump(json_game_data, file)

