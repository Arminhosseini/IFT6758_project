# Import necessary modules and classes
import pandas as pd

from crawler import Crawler
from penalty import Penalty
from team import Team

# Create an instance of the Crawler class
crawler = Crawler()


# Function to remove penalties that have expired
def remove_penalty(team: Team, time):
    latest_removal = 0
    for p in team.penalties:
        if p.end_time <= time:
            end_time = team.remove_penalty(p)
            if end_time > latest_removal:
                latest_removal = end_time
    return latest_removal


# Function to set the start time of a power play
def set_power_play(new_start, home: Team, away: Team, prev_start):
    if home.n_players != away.n_players and prev_start == 0:
        return new_start
    if home.n_players == away.n_players:
        return 0
    return prev_start


# Function to extract bonus features from the game data
def get_bonus_features(game_id):
    raw_data = crawler.read_data_by_game_id(game_id)

    # Extract relevant information from the raw data
    game_id = raw_data["gamePk"]
    home_name = raw_data["gameData"]["teams"]["home"]["name"]
    away_name = raw_data["gameData"]["teams"]["away"]["name"]

    # Initialize Team objects for the home and away teams
    home = Team(home_name)
    away = Team(away_name)

    # Initialize the start time of the power play
    power_play_start = 0
    # List to store bonus features for each play
    plays_bonus_features = list()
    # Iterate through each play in the game
    for play in raw_data["liveData"]["plays"]["allPlays"]:
        # Extract information about the play
        period = play["about"]["period"]
        period_time = play["about"]["periodTime"]
        time = (period - 1) * 20 * 60
        time += sum(
            int(x) * 60**i for i, x in enumerate(period_time.split(":")[::-1])
        )

        # Remove expired penalties from the home teams
        latest_removal = remove_penalty(home, time)
        power_play_start = set_power_play(latest_removal, home, away, power_play_start)

        # Remove expired penalties from the away teams
        latest_removal = remove_penalty(away, time)
        power_play_start = set_power_play(latest_removal, home, away, power_play_start)

        # Process plays based on their event type
        if play["result"]["event"] == "Penalty":
            # Determine the penalty team and its minutes
            team = home if play["team"]["name"] == home_name else away
            penalty_min = play["result"]["penaltyMinutes"]

            # Add penalty to the team's penalties if it's not a 10-minute or 0-minute penalty
            if penalty_min != 10 and penalty_min != 0:
                team.add_penalty(Penalty(penalty_min, time))
                # Update power play start time if the teams have different player counts
                if home.n_players != away.n_players:
                    power_play_start = time

        event_type = play["result"]["event"]
        if event_type == "Goal":
            team = home if play["team"]["name"] == home_name else away
            other_team = home if team == away else away

            # Calculate power play time and adjust based on player counts
            power_play_time = time - power_play_start if power_play_start != 0 else 0
            n_friend = team.n_players
            n_oppose = other_team.n_players

            if n_friend < n_oppose:
                power_play_time *= -1

            # Append bonus features for a goal to the plays_bonus_features list
            plays_bonus_features.append(
                (period, period_time, event_type, power_play_time, n_friend, n_oppose)
            )

            # Remove a minor penalty from the opposing team (if applicable)
            if team.n_players != other_team.n_players:
                for p in other_team.penalties:
                    if p.type == "minor":
                        latest_removal = other_team.remove_penalty(p)
                        power_play_start = set_power_play(
                            latest_removal, home, away, power_play_start
                        )
                        break
                    # Convert double minor to a regular minor if more than 2 minutes remain
                    elif p.type == "double":
                        if p.end_time - time > 120:
                            p = Penalty(2, time)
                        else:
                            latest_removal = other_team.remove_penalty(p)
                            power_play_start = set_power_play(
                                latest_removal, home, away, power_play_start
                            )
                        break

        if event_type == "Shot":
            team = home if play["team"]["name"] == home_name else away
            other_team = home if team == away else away

            # Calculate power play time and adjust based on player counts
            power_play_time = time - power_play_start if power_play_start != 0 else 0
            n_friend = team.n_players
            n_oppose = other_team.n_players

            if n_friend < n_oppose:
                power_play_time *= -1

            # Append bonus features for a shot to the plays_bonus_features list
            plays_bonus_features.append(
                (period, period_time, event_type, power_play_time, n_friend, n_oppose)
            )

    # Return the list of bonus features for each play in the game
    return plays_bonus_features


if __name__ == "__main__":
    # Read tidy data from a CSV file and preprocess it
    tidy_data = pd.read_csv("tidyData_fe3.csv", index_col=False)
    tidy_data = tidy_data.drop("Unnamed: 0", axis=1, errors="ignore")
    tidy_data[["power_play_time", "n_friend", "n_oppose"]] = None, None, None

    # Get unique game IDs from the tidy data
    game_ids = tidy_data["gamePk"].unique()
    i = 0
    for game_id in game_ids:
        print(game_id)
        # Get shots and goals events for each game_id
        plays = tidy_data[tidy_data["gamePk"] == game_id]
        plays_bonus_features = get_bonus_features(game_id)
        # An error happens if number of mentioned events are not equal in
        # previous tidy data and plays_bonus_features
        if len(plays) != len(plays_bonus_features):
            print("Error in game_id:", game_id)
            print("Number of plays:", len(plays))
            print("Number of bonus_features:", len(plays_bonus_features))
            break
        # Add new features to tidy data
        for bonus_feature in plays_bonus_features:
            tidy_data.at[i, "power_play_time"] = bonus_feature[3]
            tidy_data.at[i, "n_friend"] = bonus_feature[4]
            tidy_data.at[i, "n_oppose"] = bonus_feature[5]
            i += 1

    # Save the modified tidy data to a new CSV file
    tidy_data.to_csv("tidyData_fe4.csv")
