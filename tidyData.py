import pandas as pd
from os import listdir


def event_filter(play):
    all_plays = play["allPlays"]
    game_id = play["gamePk"]
    try:
        result = all_plays["result"]
        event = result["event"]
        return event in ["Shot", "Goal"]
    except Exception as err:
        return False


df = pd.read_json("Dataset/2016-playoffs.json")
df["allPlays"] = df["liveData"].apply(lambda x: x["plays"].get("allPlays"))
df = df[["gamePk", "allPlays"]]
df = df.explode("allPlays")
df.reset_index(drop=True, inplace=True)
mask = df.apply(event_filter, axis=1)
df = df[mask]
df.reset_index(drop=True, inplace=True)
df["period"] = df["allPlays"].apply(lambda x: x.get("about")["period"])
df["periodType"] = df["allPlays"].apply(lambda x: x.get("about")["periodType"])
df["periodTime"] = df["allPlays"].apply(lambda x: x.get("about")["periodTime"])
df["periodTimeRemaining"] = df["allPlays"].apply(
    lambda x: x.get("about")["periodTimeRemaining"])
df["dateTime"] = df["allPlays"].apply(lambda x: x.get("about")["dateTime"])
df["dateTime"] = df["allPlays"].apply(lambda x: x.get("about")["dateTime"])
df["teamId"] = df["allPlays"].apply(lambda x: x.get("team")["id"])
df["teamName"] = df["allPlays"].apply(lambda x: x.get("team")["name"])
df["teamTriCode"] = df["allPlays"].apply(lambda x: x.get("team")["triCode"])
df["eventType"] = df["allPlays"].apply(lambda x: x.get("result")["event"])
df["x-coordinate"] = df["allPlays"].apply(lambda x: x.get("coordinates")["x"])
df["y-coordinate"] = df["allPlays"].apply(lambda x: x.get("coordinates")["y"])
df["goalieName"] = df["allPlays"].apply(lambda x: ", ".join([(player.get("player")[
                                        "fullName"]) for player in x.get("players") if player.get("playerType") == "Goalie"]))

df["shooterName"] = df["allPlays"].apply(lambda x: ", ".join([(player.get("player")[
    "fullName"]) for player in x.get("players") if player.get("playerType") in ["Shooter", "Scorer"]]))

df["shotType"] = df["allPlays"].apply(
    lambda x: x.get("result")["secondaryType"])
df["isEmptyNet"] = df["allPlays"].apply(lambda x: x.get(
    "result")["emptyNet"] if x.get("result")["event"] == "Goal" else False)

df["strength"] = df["allPlays"].apply(lambda x: x.get(
    "result")["strength"]["name"] if x.get("result")["event"] == "Goal" else False)

df = df.drop(columns=['allPlays'])

df.to_csv('Dataset/tidyData.csv', index=False)
