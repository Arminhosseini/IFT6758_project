# Part 4 Feature Engineering II

#### We added the features from the list below
| Feature Name                | Explanation                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| game_second                 | Time (in seconds) of each event relative to the start of the entire game |
| last_event_type             | Type of the last event.                                                  |
| coor_x_last_event           | X-coordinate of the last event.                                          |
| coor_y_last_event           | Y-coordinate of the last event.                                          |
| time_last_event             | Time elapsed since the last event (seconds).                             |
| distance_last_event         | Distance from the last event.                                            |
| is_rebound                  | True if the last event was a rebound, otherwise False.                   |
| Change in shot angle        | Angle change in the shot, only applicable for rebounds.                  |
| Speed                       | Defined as distance from the previous event divided by time.             |
| power_play_time **(bonus)** | Time elapsed since the start of the power play (seconds).                |
| n_friend        **(bonus)** | Number of friendly non-goalie skaters on the ice.                        |
| n_oppose        **(bonus)** | Number of opposing non-goalie skaters on the ice.                        |

#### After adding the features, we processed the data. First we split the data into a training set and a test set((the function `split_train_test()` was used here), and then the subsequent processing of the data will be different depending on the model chosen. Then we removed features that were not relevant to the modeling (the function `remove_extra_features()` was used) and added a new feature `game_second`. Finally we uploaded `train.csv`, `test_regular.csv`, `test_playoff.csv` to comet. ml

## Part 4 Question 5

#### For this particular match, we wrote a separate py file `get_game.py` to run it, and the purpose of this file is to upload this match as a dataframe to comet.ml, named according to requirements(`wpg_v_wsh_2017021065.csv`)
#### [Link here!](https://www.comet.com/ift6758-b09-project/ift6758-project-milestone2/a277e307828340a78555e8fe7e38a04a)
