# Import the Penalty class from the penalty module
from penalty import Penalty


# Class definition for Team
class Team:
    def __init__(self, name):
        # Initialize Team object with a name
        self.name = name

        # Set the initial number of players to 5
        self.n_players = 5

        # Initialize lists to store current penalties and reserved penalties
        self.penalties = list()
        self.reserved_penalties = list()

    # Assign penalty to team
    def add_penalty(self, penalty: Penalty):
        # Check if there are more than 3 players on the team
        if self.n_players > 3:
            # Add the penalty to the current penalties list
            self.penalties.append(penalty)
            # Decrement the number of players
            self.n_players -= 1
        else:
            # If the team has 3 or fewer players, add the penalty to the reserved penalties list
            self.reserved_penalties.append(penalty)

    # Remove the penalty from the active penalties list
    def remove_penalty(self, penalty):
        self.penalties.remove(penalty)
        # Increase the number of players in the team
        self.n_players += 1

        # Check if there are reserved penalties
        if len(self.reserved_penalties) > 0:
            # Pop the last reserved penalty
            new_p = self.reserved_penalties.pop()
            # Add a new penalty to the active penalties list based on the popped reserved penalty
            self.penalties.append(Penalty(new_p.penalty_min, penalty.end_time))
            # Decrease the number of players in the team
            self.n_players -= 1

        # Return the end time of the removed penalty
        return penalty.end_time
