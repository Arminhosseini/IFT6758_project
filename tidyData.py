import pandas as pd
from os import listdir
from crawler_notpushed import Crawler

# Define a function to filter events and keep only "Shot" and "Goal"


class tidy_data():
    def __init__(self, data) -> None:
        self.write_path = 'Dataset/tidyData.csv'
        self.df = pd.DataFrame(data)

    def event_filter(self, play):
        all_plays = play["allPlays"]
        game_id = play["gamePk"]
        try:
            result = all_plays["result"]
            event = result["event"]
            return event in ["Shot", "Goal"]
        except Exception as err:
            return False

    # Define a function to find the home team
    def home_team(self, x):
        try:
            home = x.get("linescore").get("teams").get(
                "home").get("team").get("name")
            return home
        except:
            return None

    # Define a function to find the attacking side

    def attacking_side(self, data):
        try:
            period_number = data["period"]
            team_name = data["teamName"]
            home_team = data["homeTeam"]
            periods = data["liveData"]["linescore"]["periods"]
            for period in periods:
                if period["num"] == period_number:
                    if team_name == home_team:
                        return period["away"]["rinkSide"]
                    else:
                        return period["home"]["rinkSide"]
        except Exception as err:
            return None

    # Define a function to tidy up and preprocess the data

    def tidy_data(self):
        try:

            # Extract the "allPlays" column
            self.df["allPlays"] = self.df["liveData"].apply(
                lambda x: x["plays"].get("allPlays"))

            # Apply home_team function to find the home team names
            self.df.loc[:, "homeTeam"] = self.df["liveData"].apply(
                lambda x: self.home_team(x))

            # Explode the "allPlays" column to separate events
            self.df = self.df.explode("allPlays")
            self.df.reset_index(drop=True, inplace=True)

            # Apply the event filter to keep only "Shot" and "Goal" events
            mask = self.df.apply(self.event_filter, axis=1)
            self.df = self.df[mask]
            self.df.reset_index(drop=True, inplace=True)

            # Extract various attributes from the "allPlays" column
            self.df.loc[:, "period"] = self.df["allPlays"].apply(
                lambda x: x.get("about")["period"])

            self.df.loc[:, "periodType"] = self.df["allPlays"].apply(
                lambda x: x.get("about")["periodType"])

            self.df.loc[:, "periodTime"] = self.df["allPlays"].apply(
                lambda x: x.get("about")["periodTime"])

            self.df.loc[:, "periodTimeRemaining"] = self.df["allPlays"].apply(
                lambda x: x.get("about")["periodTimeRemaining"])

            self.df.loc[:, "dateTime"] = self.df["allPlays"].apply(
                lambda x: x.get("about")["dateTime"])

            self.df.loc[:, "teamId"] = self.df["allPlays"].apply(
                lambda x: x.get("team")["id"])

            self.df.loc[:, "teamName"] = self.df["allPlays"].apply(
                lambda x: x.get("team")["name"])

            self.df.loc[:, "attackingSide"] = self.df.apply(
                self.attacking_side, axis=1)

            self.df.loc[:, "teamTriCode"] = self.df["allPlays"].apply(
                lambda x: x.get("team")["triCode"])

            self.df.loc[:, "eventType"] = self.df["allPlays"].apply(
                lambda x: x.get("result")["event"])

            self.df.loc[:, "x-coordinate"] = self.df["allPlays"].apply(lambda x: x.get("coordinates")["x"]
                                                                       if not len(x.get("coordinates")) == 0 and 'x' in x.get("coordinates") else None)

            self.df.loc[:, "y-coordinate"] = self.df["allPlays"].apply(lambda x: x.get("coordinates")["y"]
                                                                       if not len(x.get("coordinates")) == 0 and 'y' in x.get("coordinates") else None)

            self.df.loc[:, "goalieName"] = self.df["allPlays"].apply(lambda x: ", ".join([(player.get("player")["fullName"])
                                                                                          for player in x.get("players") if player.get("playerType") == "Goalie"]))

            self.df.loc[:, "shooterName"] = self.df["allPlays"].apply(lambda x: ", ".join([(player.get("player")["fullName"])
                                                                                           for player in x.get("players") if player.get("playerType") in ["Shooter", "Scorer"]]))

            self.df.loc[:, "shotType"] = self.df["allPlays"].apply(lambda x: x.get("result")["secondaryType"]
                                                                   if "secondaryType" in x.get("result") else None)

            self.df.loc[:, "isEmptyNet"] = self.df["allPlays"].apply(lambda x: x.get("result")["emptyNet"]
                                                                     if x.get("result")["event"] == "Goal" and "emptyNet" in x.get("result") else None)

            self.df.loc[:, "strength"] = self.df["allPlays"].apply(lambda x: x.get("result")["strength"]["name"]
                                                                   if x.get("result")["event"] == "Goal" else False)

            # Drop the "unnecessary" column
            self.df = self.df.drop(columns=['allPlays', 'copyright',
                                            'link', 'metaData', 'gameData', 'liveData'])

            # Return the cleaned and tidied data
            return self.df

        except Exception as err:
            print("Error in tidy data: {}".format(err))

    def write(self, is_first):
        if is_first:
            self.df.to_csv(self.write_path, mode='a', index=False, header=True)
        else:
            self.df.to_csv(self.write_path, mode='a',
                           index=False, header=False)


if __name__ == "__main__":
    # Iterate over JSON files in the 'Dataset' directory and tidy each one
    crawler = Crawler()
    df = crawler.read_data()
    is_first = True
    for season, season_data in df.items():
        for type, type_data in season_data.items():
            new_list = list()
            for game_id, data in type_data.items():
                new_list.append(data)

            tidy = tidy_data(new_list)
            tidy.tidy_data()
            tidy.write(is_first)
            is_first = False
