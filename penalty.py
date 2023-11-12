# A dictionary mapping penalty durations to their corresponding types
min_to_type = {"2": "minor", "4": "double", "5": "major"}


# Class definition for Penalty
class Penalty:
    def __init__(self, penalty_min, game_time):
        # Initialize the Penalty object with penalty duration and game time
        self.penalty_min = penalty_min
        # Determine the penalty type based on the duration
        self.type = min_to_type[str(penalty_min)]
        # Calculate the end time of the penalty by adding the penalty duration
        self.end_time = game_time + penalty_min * 60
