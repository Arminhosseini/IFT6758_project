from pathlib import Path
import requests
import json


class Crawler:
    def __init__(self):
        self.base_url = "https://statsapi.web.nhl.com/api/v1/"
        self.dataset_path = Path("Dataset")
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.game_types = {
            "preseason": "01",
            "regular_season": "02",
            "playoffs": "03",
            "all-star": "04",
        }

        self.data = dict()

    def get_regular_game_id(self, season, number):
        number = str(number).zfill(4)
        season = str(season)
        return season + self.game_types["regular_season"] + number

    def get_playoff_game_id(self, season, round, matchup, game):
        number = f"0{round}{matchup}{game}"
        season = str(season)
        return season + self.game_types["playoffs"] + number

    def get_url(self, game_id):
        return self.base_url + f"game/{game_id}/feed/live"

    def get_game_data(self, game_id):
        game_url = self.get_url(game_id)
        # print(game_url)
        response = requests.get(game_url)
        if response.status_code != 200:
            raise Exception("404 Page Not Found")
        # return json.loads(requests.get(game_url).text)
        return response.json()

    def get_regular_data(self, season):
        data = list()
        game_number = 1
        while True:
            try:
                game_id = self.get_regular_game_id(season, game_number)
                game_data = self.get_game_data(game_id)
                data.append(game_data)
            except Exception as e:
                if str(e) == "404 Page Not Found":
                    break
            game_number += 1
        return data

    def get_playoff_data(self, season):
        data = list()
        round, matchup, game = 1, 1, 1
        while True:
            try:
                game_id = self.get_playoff_game_id(season, round, matchup, game)
                game_data = self.get_game_data(game_id)
                data.append(game_data)
            except Exception as e:
                if str(e) == "404 Page Not Found":
                    matchup += 1
                    game = 1
                    try:
                        game_id = self.get_playoff_game_id(season, round, matchup, game)
                        game_data = self.get_game_data(game_id)
                        data.append(game_data)
                    except Exception as e:
                        if str(e) == "404 Page Not Found":
                            round += 1
                            matchup = 1
                            game = 1
                            try:
                                game_id = self.get_playoff_game_id(
                                    season, round, matchup, game
                                )
                                game_data = self.get_game_data(game_id)
                                data.append(game_data)
                            except Exception as e:
                                if str(e) == "404 Page Not Found":
                                    break
            game += 1
        return data

    def get_total_data(self, start_season=2016, end_season=2020):
        for season in range(start_season, end_season + 1):
            self.data[season] = dict()
            self.data[season]["regular_season"] = self.get_regular_data(season)
            self.data[season]["playoffs"] = self.get_playoff_data(season)

    def write_data(self):
        for season, season_data in self.data.items():
            for type, data in season_data.items():
                file_path = self.dataset_path / f"{season}-{type}.json"
                with open(file_path, "w") as file:
                    json.dump(data, file)

    def crawl(self):
        self.get_total_data()
        self.write_data()


if __name__ == "__main__":
    c = Crawler()
    c.crawl()