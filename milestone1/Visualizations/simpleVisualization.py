import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# get the tidy data path
TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), 'Dataset', 'tidyData.csv')

# save the figures
SAVE_FIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))


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
        plt.xticks(rotation=20)
        plt.ylabel('Count of Shots')
        plt.xlabel('Type of Shot')
        plt.title('Shot & Goal Count Per Type of Shot and Percentage of Successful Goals for 2016-2017 season')
        ax.legend(labels=['Shots', 'Goals'], loc='upper right')

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
                    size=12, ha="center"
                )

        figures_dir = 'figures'
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

            # Filter out rows with null x or y coordinates in goal_events_df
            shot_events_df = shot_events_df.dropna(subset=['x-coordinate', 'y-coordinate'])

            def compute_shot_distance(x, y):
                x = float(x)  # Convert x-coordinate to float
                y = float(y)  # Convert y-coordinate to float
                goal_position = (89, 0) if x > 0 else (-89, 0)
                return np.sqrt((x - goal_position[0]) ** 2 + (y - goal_position[1]) ** 2)

            # Add 'shot_distance' column to goal_events_df
            shot_events_df['shot_distance'] = shot_events_df.apply(
                lambda row: compute_shot_distance(row['x-coordinate'], row['y-coordinate']), axis=1)
            shot_events_df['shot_distance'] = shot_events_df['shot_distance'].astype(float)

            # Filter the DataFrame to include 'eventType' = 'Goal'
            shot_events_df['is_goal'] = shot_events_df['eventType'].apply(lambda x: x == 'Goal')

            # Group the data by shot_distance and calculate the mean goal probability
            grouped_df = shot_events_df.groupby("shot_distance")["is_goal"].mean().reset_index()
            # max_shot_distance = grouped_df['shot_distance'].max()
            # print(max_shot_distance)
            # Create a figure and axis for the plot
            fig = plt.figure(figsize=(10, 6))
            ax = sns.lineplot(x='shot_distance', y='is_goal', data=grouped_df)

            # Customize the plot
            ax.set_title(f'Shot Distance vs Goal Chance ({season}-{int(season) + 1})')
            ax.set_xlabel('Shot Distance (feet)')
            ax.set_ylabel('Average Goal Chance')
            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed')
            # Create the directory for saving figures if it does not exist
            figures_dir = 'figures'
            os.makedirs(figures_dir, exist_ok=True)

            # Save the figure if requested
            if save_fig:
                fig.savefig(os.path.join(figures_dir, f'Q5-2_shot_distance_vs_goal_chance_{season}.png'))

            figures.append(fig)

        # Show the plots after generating all figures
        plt.show()

        return figures

    # Part 5 Question 3
    def distance_and_type_vs_goal(self, df, save_fig: bool = True) -> list[plt.Figure]:
        """
        Create line plots showing the relationship between shot distance and goal percentage for each shot type.
        :param df: Tidy pandas.DataFrame (contains data for a specific season)
        :param save_fig: Boolean to save the plots to SAVE_FIG_PATH
        :return: List of plt.Figure object instances
        """
        # Filter the DataFrame to include only rows for the season
        season_df = df[df['gamePk'].astype(str).str.startswith('2018')].copy()

        # Filter out rows with NaN in 'shotType'
        season_df = season_df.dropna(subset=['shotType'])

        # Encode 'shotType' column to numerical values
        season_df['shotType'] = season_df['shotType'].astype('category')
        season_df['shotTypeCode'] = season_df['shotType'].cat.codes

        def compute_shot_distance(x, y):
            x = float(x)  # Convert x-coordinate to float
            y = float(y)  # Convert y-coordinate to float
            goal_position = (89, 0) if x > 0 else (-89, 0)
            return np.sqrt((x - goal_position[0]) ** 2 + (y - goal_position[1]) ** 2)

        # Add 'shot_distance' column to goal_events_df
        season_df['shot_distance'] = season_df.apply(
            lambda row: compute_shot_distance(row['x-coordinate'], row['y-coordinate']), axis=1)
        season_df['shot_distance'] = season_df['shot_distance'].astype(float)

        # Calculate 'is_goal' based on 'eventType'
        season_df['is_goal'] = season_df['eventType'] == 'Goal'

        # Get unique shot types
        unique_shot_types = season_df['shotType'].cat.categories

        # Initialize an empty list to store the figures
        figures = []

        for shot_type in unique_shot_types:
            # Filter the data for the current shot type
            shot_type_df = season_df[season_df['shotType'] == shot_type]

            # Group the data by both 'shotType' and 'shot_distance'
            grouped_df = shot_type_df.groupby(['shotType', 'shot_distance'], observed=False)['is_goal'].mean().reset_index()

            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=self.fig_size)

            # Plot the line for the current shot type
            sns.lineplot(x='shot_distance', y='is_goal', data=grouped_df, ax=ax, label=shot_type)

            # Customize the plot
            ax.set_title(f'Goal Percentage by Shot Distance for {shot_type} Shots (2018-2019 Season)')
            ax.set_xlabel('Shot Distance (feet)')
            ax.set_ylabel('Average Goal Percentage')
            ax.legend()

            # Save the figure if requested
            if save_fig:
                figures_dir = 'figures'
                os.makedirs(figures_dir, exist_ok=True)
                fig_path = os.path.join(figures_dir, f'Q5-3_distance_and_type_vs_goal_{shot_type}.png')
                plt.savefig(fig_path)

            figures.append(fig)

        # Show the plots after generating all figures
        plt.show()

        return figures
if __name__ == "__main__":

    # Read the DataFrame
    df = pd.read_csv(TIDY_DATA_PATH, low_memory=False)
    #print(df.columns)
    hockey_figure = HockeyFigure()
    #hockey_figure.shot_type_histogram(df)
    #hockey_figure.create_distance_vs_goal_chance_plot(df)
    hockey_figure.distance_and_type_vs_goal(df)

