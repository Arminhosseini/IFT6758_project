import pandas as pd
from os import listdir

# Define a function to filter events and keep only "Shot" and "Goal"

# Define a function to filter events and keep only "Shot" and "Goal"


def event_filter(play):
    all_plays = play["allPlays"]
    game_id = play["gamePk"]
    try:
        result = all_plays["result"]
        event = result["event"]
        return event in ["Shot", "Goal"]
    except Exception as err:
        return False

# Define a function to tidy up and preprocess the data


def tidy_data(path):
    try:
        # Read the JSON file into a DataFrame
        df = pd.read_json('Dataset/' + path)

        # Extract the "allPlays" column
        df["allPlays"] = df["liveData"].apply(
            lambda x: x["plays"].get("allPlays"))

        # Keep only the relevant columns
        df = df[["gamePk", "allPlays"]]

        # Explode the "allPlays" column to separate events
        df = df.explode("allPlays")
        df.reset_index(drop=True, inplace=True)

        # Apply the event filter to keep only "Shot" and "Goal" events
        mask = df.apply(event_filter, axis=1)
        df = df[mask]
        df.reset_index(drop=True, inplace=True)

        # Extract various attributes from the "allPlays" column
        df.loc[:, "period"] = df["allPlays"].apply(
            lambda x: x.get("about")["period"])
        df.loc[:, "periodType"] = df["allPlays"].apply(
            lambda x: x.get("about")["periodType"])
        df.loc[:, "periodTime"] = df["allPlays"].apply(
            lambda x: x.get("about")["periodTime"])
        df.loc[:, "periodTimeRemaining"] = df["allPlays"].apply(
            lambda x: x.get("about")["periodTimeRemaining"])
        df.loc[:, "dateTime"] = df["allPlays"].apply(
            lambda x: x.get("about")["dateTime"])
        df.loc[:, "teamId"] = df["allPlays"].apply(
            lambda x: x.get("team")["id"])
        df.loc[:, "teamName"] = df["allPlays"].apply(
            lambda x: x.get("team")["name"])
        df.loc[:, "teamTriCode"] = df["allPlays"].apply(
            lambda x: x.get("team")["triCode"])
        df.loc[:, "eventType"] = df["allPlays"].apply(
            lambda x: x.get("result")["event"])
        df.loc[:, "x-coordinate"] = df["allPlays"].apply(lambda x: x.get("coordinates")["x"]
                                                         if not len(x.get("coordinates")) == 0 and 'x' in x.get("coordinates") else None)
        df.loc[:, "y-coordinate"] = df["allPlays"].apply(lambda x: x.get("coordinates")["y"]
                                                         if not len(x.get("coordinates")) == 0 and 'y' in x.get("coordinates") else None)
        df.loc[:, "goalieName"] = df["allPlays"].apply(lambda x: ", ".join([(player.get("player")["fullName"])
                                                                            for player in x.get("players") if player.get("playerType") == "Goalie"]))
        df.loc[:, "shooterName"] = df["allPlays"].apply(lambda x: ", ".join([(player.get("player")["fullName"])
                                                                             for player in x.get("players") if player.get("playerType") in ["Shooter", "Scorer"]]))
        df.loc[:, "shotType"] = df["allPlays"].apply(lambda x: x.get("result")["secondaryType"]
                                                     if "secondaryType" in x.get("result") else None)
        df.loc[:, "isEmptyNet"] = df["allPlays"].apply(lambda x: x.get("result")["emptyNet"]
                                                       if x.get("result")["event"] == "Goal" and "emptyNet" in x.get("result") else None)
        df.loc[:, "strength"] = df["allPlays"].apply(lambda x: x.get("result")["strength"]["name"]
                                                     if x.get("result")["event"] == "Goal" else False)

        # Drop the "allPlays" column
        df = df.drop(columns=['allPlays'])

        # Save the cleaned and tidied data to a CSV file
        df.to_csv('Dataset/tidyData.csv', mode='a', index=False)

    except ValueError as e:
        print(f"Error reading JSON file {path}: {e}")


# Iterate over JSON files in the 'Dataset' directory and tidy each one
for file in listdir('Dataset'):
    tidy_data(file)
