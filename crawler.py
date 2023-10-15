# Import necessary modules for file manipulation, web requests, and JSON handling
from pathlib import Path
import requests
import json


# Define a class called Crawler for NHL game data extraction and storage
class Crawler:
    # Initialize instance variables
    def __init__(self):
        # Set the base URL for NHL game data API
        self.base_url = "https://statsapi.web.nhl.com/api/v1/"

        # Create a directory for the dataset if it doesn't exist
        self.dataset_path = Path("Dataset")
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Define a path for the dataset information in JSON format
        self.dataset_info_path = self.dataset_path / Path("dataset_info.json")

        # Define dictionaries for different game types and their corresponding codes
        self.game_types = {
            "preseason": "01",
            "regular_season": "02",
            "playoffs": "03",
            "all-star": "04",
        }

        # Create an inverted dictionary to map codes to game types
        self.inv_game_types = {v: k for k, v in self.game_types.items()}

        # Initialize data storage dictionaries
        self.data = dict()
        self.dataset_info = dict()

        # Set season range for data collection based on the dataset_info file
        self.dataset_start_season = None
        self.dataset_end_season = None
        if self.dataset_info_path.is_file():
            dataset_info = json.loads(open(self.dataset_info_path, "r").read())
            self.dataset_start_season = dataset_info["start_season"]
            self.dataset_end_season = dataset_info["end_season"]

    # Define functions for generating game IDs in regular games
    # based on season and game number
    def get_regular_game_id(self, season, number):
        number = str(number).zfill(4)
        season = str(season)
        return season + self.game_types["regular_season"] + number

    # Define functions for generating game IDs in playoff games
    # based on season and game number
    def get_playoff_game_id(self, season, round, matchup, game):
        number = f"0{round}{matchup}{game}"
        season = str(season)
        return season + self.game_types["playoffs"] + number

    # Generate the full URL for fetching game data using a game ID
    def get_url(self, game_id):
        return self.base_url + f"game/{game_id}/feed/live"

    # Fetch game data from the NHL API for a given game ID
    def get_game_data(self, game_id):
        game_url = self.get_url(game_id)
        print(game_url)
        response = requests.get(game_url)
        # Check if the response status code indicates a successful request
        if response.status_code != 200:
            # Raise an exception indicating that the page was not found (404)
            raise Exception("404 Page Not Found")
        return json.loads(requests.get(game_url).text)

    # Functions to retrieve data for regular season
    def get_regular_data(self, season):
        data = list()
        game_number = 1
        # Continue looping until there are no more games found
        while True:
            try:
                # Generate a unique game ID for the current season and game number
                game_id = self.get_regular_game_id(season, game_number)
                # Fetch game data from the NHL API using the game ID
                game_data = self.get_game_data(game_id)
                data.append(game_data)
            except Exception as e:
                if str(e) == "404 Page Not Found":
                    # It means there are no more games for this season
                    break
            game_number += 1
        return data

    # Functions to retrieve data for playoffs
    def get_playoff_data(self, season):
        data = list()
        round, matchup, game = 1, 1, 1
        while True:
            try:
                # Generate a unique game ID for the current season and game number
                game_id = self.get_playoff_game_id(season, round, matchup, game)
                # Fetch game data from the NHL API using the game ID
                game_data = self.get_game_data(game_id)
                data.append(game_data)
            except Exception as e:
                if str(e) == "404 Page Not Found":
                    # If a game is not found, increment the matchup and reset the game number
                    matchup += 1
                    game = 1
                    try:
                        # Generate a unique game ID for the current season and game number
                        game_id = self.get_playoff_game_id(season, round, matchup, game)
                        # Fetch game data from the NHL API using the game ID
                        game_data = self.get_game_data(game_id)
                        data.append(game_data)
                    except Exception as e:
                        if str(e) == "404 Page Not Found":
                            # If a game is not found, increment the round and reset the game number and matchup
                            round += 1
                            matchup = 1
                            game = 1
                            try:
                                # Generate a unique game ID for the current season and game number
                                game_id = self.get_playoff_game_id(
                                    season, round, matchup, game
                                )
                                # Fetch game data from the NHL API using the game ID
                                game_data = self.get_game_data(game_id)
                                data.append(game_data)
                            except Exception as e:
                                if str(e) == "404 Page Not Found":
                                    # If no game is found for the next round, exit the loop
                                    break
            # Increment the game number to move to the next game in the current matchup
            game += 1
        return data

    # Get data for a specified range of seasons and store it in self.data
    def get_total_data(self, start_season=2016, end_season=2020):
        # Set the start and end seasons in the dataset information
        self.dataset_info["start_season"] = start_season
        self.dataset_info["end_season"] = end_season

        # Loop through the range of seasons from start_season to end_season (inclusive)
        for season in range(start_season, end_season + 1):
            self.data[season] = dict()
            self.data[season]["regular_season"] = self.get_regular_data(season)
            self.data[season]["playoffs"] = self.get_playoff_data(season)

    # Write the collected data to JSON files in the Dataset directory
    def write_data(self):
        for season, season_data in self.data.items():
            for type, data in season_data.items():
                # Create a directory path for the current season and game type
                file_dir = self.dataset_path.joinpath(str(season), str(type))
                file_dir.mkdir(parents=True, exist_ok=True)
                # Write the game data in JSON format to the file
                for game_data in data:
                    file_name = f"{game_data['gamePk']}.json"
                    with open(file_dir / file_name, "w") as file:
                        json.dump(game_data, file)

    # Write dataset information to a JSON file
    def write_dataset_info(self):
        with open(self.dataset_info_path, "w") as file:
            json.dump(self.dataset_info, file)

    # Read all the collected data from JSON files into a structured dictionary
    def read_data(self):
        data = dict()
        # Loop through all JSON files in the dataset directory and its subdirectories
        for game_data_path in self.dataset_path.glob("**/**/*.json"):
            # Continue if the path is related to dataset information
            if "dataset_info" in str(game_data_path):
                continue

            # Extract information from the file path
            tokens = str(game_data_path).split(".")[0].split("/")[1:4]
            season, type, game_id = int(tokens[0]), tokens[1], int(tokens[2])

            # Load and store the game data from the JSON file
            if season not in data:
                data[season] = dict()
            if type not in data[season]:
                data[season][type] = dict()
            data[season][type][game_id] = json.loads(open(game_data_path, "r").read())

        return data

    # Read data for a specific game by its game ID
    def read_data_by_game_id(self, game_id):
        # Extract information from the game ID
        game_id_str = str(game_id)
        season = game_id_str[:4]
        type = self.inv_game_types[game_id_str[4:6]]

        # Construct the file path to the JSON file containing the game data
        file_path = self.dataset_path.joinpath(season, type, f"{game_id_str}.json")

        # Read the contents of the JSON file, and return the game data as a dictionary
        return json.loads(open(file_path, "r").read())

    # The main method to initiate data collection and storage
    def crawl(self):
        self.get_total_data()
        self.write_data()
        self.write_dataset_info()


if __name__ == "__main__":
    c = Crawler()
    c.crawl()
