import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# get the tidy data path
TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), 'Dataset', 'tidyData.csv')

# save the figures
SAVE_FIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'images', 'simpleVisualization'))


# Define a function to compute shot distance based on attackingSide
def compute_shot_distance(row):
    x = float(row['x-coordinate'])
    y = float(row['y-coordinate'])
    attacking_side = row['attackingSide']

    if attacking_side == 'left':
        goal_position = (-89, 0)
    elif attacking_side == 'right':
        goal_position = (89, 0)
    else:
        raise ValueError("Invalid value in 'attackingSide' column")

    return np.sqrt((x - goal_position[0]) ** 2 + (y - goal_position[1]) ** 2)


class HockeyFigure:
    def __init__(self, fig_size=(20, 15)):
        self.fig_size = fig_size

    # Part 5 Question 1
    def shot_type_histogram(self, df, save_fig: bool = True) -> plt.Figure:
        """
        Displays a shot-type histogram as described in Part 5 Question 1
        :param df: tidy pandas.DataFrame ( I chose the 2016 season here)
        :param save_fig: boolean to save the plot to SAVE_FIG_PATH
        :return: a plt.Figure object instance
        """

        # Filter df to only include games from 2016
        df_2016 = df[df['gamePk'].astype(str).str.startswith('2016')]

        # Create a new column 'is_goal' based on 'eventType' to determine if it's a goal
        df_2016 = df_2016.copy()
        df_2016['is_goal'] = df_2016['eventType'] == 'Goal'

        # Filter out rows with NaN in 'shotType'
        df_2016 = df_2016.dropna(subset=['shotType'])

        # Create a figure
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Get unique shot types
        unique_shot_types = df_2016['shotType'].unique()

        # Plot shot type counts with hue='is_goal'
        sns.countplot(x='shotType',
                      data=df_2016,
                      order=unique_shot_types,
                      hue='is_goal',
                      legend='full',
                      palette=['#FF0000', '#0000FF'])

        # Customize the plot
        plt.xticks(rotation=20, fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Count of Shots', fontdict={'size': 16})
        plt.xlabel('Type of Shot', fontdict={'size': 16})
        plt.title('Shot & Goal Count Per Type of Shot and Percentage of Successful Goals for 2016-2017 season',
                  fontdict={'size': 20})
        ax.legend(labels=['Shots', 'Goals'], loc='upper right', fontsize=16)

        # Calculate and add goal and shot counts and percentages on top of the bars
        for idx, p in enumerate(ax.patches):
            height = p.get_height()
            if idx < len(unique_shot_types):  # Ensure index does not exceed unique_shot_types length
                shot_type = unique_shot_types[idx]
                shot_count = df_2016[df_2016['shotType'] == shot_type]['is_goal'].count()
                goal_count = df_2016[(df_2016['shotType'] == shot_type) & (df_2016['is_goal'] == True)][
                    'is_goal'].count()
                percentage_goals = (goal_count / shot_count) * 100

                ax.text(
                    p.get_x() + p.get_width() / 2., height + 30,
                    f'Shots: {shot_count}\nGoals: {goal_count}\nPercentage: {percentage_goals:.2f}%',
                    size=16, ha="center"
                )

        figures_dir = SAVE_FIG_PATH
        os.makedirs(figures_dir, exist_ok=True)
        # Save the figure if requested
        if save_fig:
            fig.savefig(os.path.join(figures_dir, f'Q5-1_shot_type_histogram.png'))

        plt.show()

        return fig

    # Part 5 Question 2
    def create_distance_vs_goal_chance_plot(self, df, save_fig: bool = True) -> list[plt.Figure]:
        """
        Plots comparative graphs for different seasons (2018-2019, 2019-2020, 2020-2021)
        of the relationship between shot distance and goals (as described in Part 5 Q2)
        :param df: DataFrame containing shot and goal data
        :param save_fig: boolean to save the plots to SAVE_FIG_PATH
        :return: a list of plt.Figure object instances
        """

        # Initialize an empty list to store the figures
        figures = []

        # Define the seasons
        seasons = ['2018', '2019', '2020']

        for season in seasons:
            # Filter the DataFrame to include only rows for the current season
            season_df = df[df['gamePk'].astype(str).str.startswith(season)].copy()

            # Filter the DataFrame to include 'eventType' = 'Shot' & 'Goal'
            shot_df = season_df[season_df['eventType'] == 'Shot']
            goal_df = season_df[season_df['eventType'] == 'Goal']
            shot_events_df = pd.concat([shot_df, goal_df], ignore_index=True)

            # Filter out rows with null x or y coordinates and non-null attackingSide
            shot_events_df = shot_events_df.dropna(subset=['x-coordinate', 'y-coordinate', 'attackingSide'])

            # Add 'shot_distance' column to shot_events_df based on attackingSide
            shot_events_df['shot_distance'] = shot_events_df.apply(compute_shot_distance, axis=1)
            shot_events_df['shot_distance'] = shot_events_df['shot_distance'].astype(float)

            # Filter the DataFrame to include 'eventType' = 'Goal'
            shot_events_df['is_goal'] = shot_events_df['eventType'].apply(lambda x: x == 'Goal')

            # Perform smoothing (e.g., rounding to nearest integer)
            shot_events_df['smoothed_shot_distance'] = shot_events_df['shot_distance'].round(0)
            total_num_shots = shot_events_df.shape[0]

            # Group the data by smoothed_shot_distance and calculate the mean goal probability
            smoothed_grouped_df = shot_events_df.groupby("smoothed_shot_distance")['is_goal'].count().reset_index()
            smoothed_grouped_df['is_goal'] = smoothed_grouped_df['is_goal'] / total_num_shots
            # print(smoothed_grouped_df)
            # Create a figure and axis for the plot
            fig = plt.figure(figsize=(10, 6))
            ax = sns.lineplot(x='smoothed_shot_distance', y='is_goal', data=smoothed_grouped_df)

            # Customize the plot
            ax.set_title(f'Shot Distance vs Goal Chance ({season}-{int(season) + 1})')
            ax.set_xlabel('Shot Distance (feet)')
            ax.set_ylabel('Average Goal Chance')
            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed')
            ax.set_xlim(0, 200)

            # Set x-axis ticks to be at intervals of 10
            plt.xticks(np.arange(0, 210, 10))

            # Create the directory for saving figures if it does not exist
            figures_dir = SAVE_FIG_PATH
            os.makedirs(figures_dir, exist_ok=True)

            # Save the figure if requested
            if save_fig:
                fig.savefig(os.path.join(figures_dir, f'Q5-2_shot_distance_vs_goal_chance_{season}.png'))

            figures.append(fig)

        # Show the plots after generating all figures
        plt.show()

        return figures

    # Part 5 Question 3
    def distance_and_type_vs_goal(self, df, save_fig: bool = True) -> plt.figure:
        """
        Create line plots showing the relationship between shot distance and goal percentage for each shot type.
        :param df: Tidy pandas.DataFrame (contains data for a specific season)
        :param save_fig: Boolean to save the plots to SAVE_FIG_PATH
        :return: A single plt.Figure object instance containing subplots for each shot type
        """
        # Filter the DataFrame to include only rows for the season
        season_df = df[df['gamePk'].astype(str).str.startswith('2016')].copy()

        # Filter out rows with NaN in 'shotType'
        season_df = season_df.dropna(subset=['shotType'])

        # Filter out rows with null x or y coordinates and non-null attackingSide
        season_df = season_df.dropna(subset=['x-coordinate', 'y-coordinate', 'attackingSide'])

        # Encode 'shotType' column to numerical values
        season_df['shotType'] = season_df['shotType'].astype('category')
        season_df['shotTypeCode'] = season_df['shotType'].cat.codes

        # Add 'shot_distance' column to goal_events_df using the compute_shot_distance function
        season_df['shot_distance'] = season_df.apply(compute_shot_distance, axis=1)
        season_df['shot_distance'] = season_df['shot_distance'].astype(float)

        # Perform smoothing (e.g., rounding to nearest integer)
        season_df['smoothed_shot_distance'] = season_df['shot_distance'].round(0)

        # Calculate 'is_goal' based on 'eventType'
        season_df['is_goal'] = season_df['eventType'] == 'Goal'

        # Get unique shot types
        unique_shot_types = season_df['shotType'].cat.categories

        # Create a figure and axis for subplots
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define a colormap with unique colors for each shot type
        colormap = matplotlib.colormaps['tab10']

        for i, shot_type in enumerate(unique_shot_types):
            # Filter the data for the current shot type
            shot_type_df = season_df[season_df['shotType'] == shot_type]
            total_num_shots = shot_type_df.shape[0]

            # Group the data by both 'shotType' and 'smoothed_shot_distance'
            grouped_df = shot_type_df.groupby(['shotType', 'smoothed_shot_distance'], observed=False)[
                'is_goal'].count().reset_index()

            grouped_df['is_goal'] = grouped_df['is_goal'] / total_num_shots
            # Plot the line for the current shot type on the same subplot with a unique color
            color = colormap(i)
            sns.lineplot(x='smoothed_shot_distance', y='is_goal', data=grouped_df, ax=ax, label=shot_type, color=color)

        # Set plot labels and limits
        ax.set_title('Goal Percentage by Shot Distance for Different Shot Types')
        ax.set_xlabel('Shot Distance (feet)')
        ax.set_ylabel('Average Goal Percentage')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 0.05)

        # Set x-axis ticks to be at intervals of 10
        plt.xticks(np.arange(0, 210, 10))

        # Save the figure if requested
        if save_fig:
            figures_dir = SAVE_FIG_PATH
            os.makedirs(figures_dir, exist_ok=True)
            fig_path = os.path.join(figures_dir, f'Q5-3_distance_and_type_vs_goal_for_2016-2017.png')
            plt.savefig(fig_path)

        # Show the combined plot
        plt.show()

        return fig


if __name__ == "__main__":
    # Read the DataFrame
    df = pd.read_csv(TIDY_DATA_PATH, low_memory=False)
    hockey_figure = HockeyFigure()
    hockey_figure.shot_type_histogram(df)
    hockey_figure.create_distance_vs_goal_chance_plot(df)
    hockey_figure.distance_and_type_vs_goal(df)
