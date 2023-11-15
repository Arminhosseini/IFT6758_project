import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comet_ml import Experiment
from dotenv import load_dotenv
import copy
import os
import seaborn as sns
load_dotenv()


def experiment_init():
    exp = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='ift6758-project-milestone2',
        workspace='ift6758-b09-project'
    )
    return exp


def compute_shot_distance(row):
    x = float(row['x-coordinate'])
    y = float(row['y-coordinate'])
    attacking_side = row['attackingSide']

    if attacking_side == 'left':
        goal_position = (-89, 0)
    elif attacking_side == 'right':
        goal_position = (89, 0)
    else:
        return np.nan

    return np.sqrt((x - goal_position[0]) ** 2 + (y - goal_position[1]) ** 2)


class feature_engineering():
    """
    A class to create extract features.

    ...

    Attributes
    ----------
    data : pd.DataFrame
        dataframe that we are working with

    """

    def __init__(self) -> None:
        """
        Initializes a FeatureEngineering1 object.

        Args:
        None

        Returns:
        None
        """
        self.data = None
        pass

    def save_data(self, path: str) -> pd.DataFrame:
        """
        Saves the data with the new features in the specified path

        Parameters
        ----------
        path : str
            The path that data should be saved to

        Returns
        -------
        None 
        """
        self.data.to_csv(path, index=False, header=True)

    def read_data(self, path: str) -> pd.DataFrame:
        """
        Reads the data from the provided path

        Parameters
        ----------
        path : str
            The path that data should be read from

        Returns
        -------
        data: pd.DataFrame
            The data with the .csv format existed in the specified path 
        """
        # read data from csv file into a dataframe
        self.data = pd.read_csv(path)

    def get_data(self) -> pd.DataFrame:
        """
        Returns the data stored in the object.

        Returns:
        pd.DataFrame: The data stored in the object.
        """
        return self.data

    def distance_from_net(self) -> None:
        """
        Measures the distance of a shot from the net

        Parameters
        ----------
        None

        Returns
        -------
        None 
        """
        self.data['shot_distance'] = self.data.apply(
            compute_shot_distance, axis=1)
        self.data['shot_distance'] = self.data['shot_distance'].astype(float)

    def angle_from_net(self) -> None:
        """
        Measures the angle of a shot in radians with respect to the net

        Parameters
        ----------
        None

        Returns
        -------
        None 
        """

        self.data['angle'] = np.arcsin(
            self.data['y-coordinate']/self.data['shot_distance'])

    def empty_net(self):
        """
        Measures if the net is empty

        Parameters
        ----------
        None

        Returns
        -------
        None 
        """
        self.data['isEmptyNet'] = self.data['isEmptyNet'].apply(
            lambda x: 1 if x == True else 0)

    def is_goal(self):
        """
        Measures if the shot is a goal

        Parameters
        ----------
        None

        Returns
        -------
        None 
        """
        self.data['isgoal'] = self.data['eventType'].apply(
            lambda x: 1 if x == 'Goal' else 0)

    def shot_counts_histogram(self, bin_type: str, path: str, save: bool = False):
        """
        Plots the histogram of shot counts (goals and no-goals seperated), binned by distance or angle

        Parameters
        ----------
        bin_type: str
            The type of binning to be used for the histogram (distance or angle)
        path: str
            The path to save the image
        save: bool
            Whether to save the image or not

        Returns
        -------
        None 
        """

        if bin_type == 'distance':
            # plot histogram of shot counts binned by distance
            p = sns.displot(
                self.data, x=self.data['shot_distance'], hue=self.data['isgoal'], bins=30, multiple="stack")
            # set axis labels and title
            p.set_axis_labels("Distance from net", "Number of shots")
            p.fig.suptitle("Distribution of shots by distance from net")
            # set figure size
            p.fig.set_size_inches(10, 5)
            plt.show()

        elif bin_type == 'angle':
            # plot histogram of shot counts binned by angle
            p = sns.displot(
                self.data, x=self.data['angle'], hue=self.data['isgoal'], bins=30, multiple="stack")
            # set axis labels and title
            p.set_axis_labels("angle from net", "Number of shots")
            p.fig.suptitle("Distribution of shots by angle from net")
            # set figure size
            p.fig.set_size_inches(10, 5)
            plt.show()
        else:
            print('Invalid bin type. Please enter either distance or angle.')

        if save:
            p.savefig(path)

    def angle_distance_joinplot(self, path: str, save: bool = False, kind: str = 'scatter'):
        """
        Plots the jointplot of angle vs distance from net

        Parameters
        ----------
        path: str
            The path to save the image
        save: bool
            Whether to save the image or not
        kind: str
            The type of plot to be used for the jointplot (default: scatter)

        Returns
        -------
        None 
        """
        p = sns.jointplot(
            data=self.data, x=self.data['shot_distance'], y=self.data['angle'], kind=kind)
        p.set_axis_labels("Distance from net", "angle from net")
        p.fig.suptitle(
            "Distribution of shots based on distance from net and angle from net", y=1.05)
        p.fig.set_size_inches(10, 10)
        plt.show()

        if save:
            p.savefig(path)

    def plot_goal_rate(self, path: str, key: str, save: bool = False):
        """
        Plots the goal percentage for each distance from net

        Parameters
        ----------
        path: str
            The path to save the image
        key: str
            The variable to plot the goal rate with respect to (distance or angle)
        save: bool
            Whether to save the image or not


        Returns
        -------
        None 
        """
        # make a copy of the data
        df = copy.deepcopy(self.data)
        # compute goal rate with respect to the variable
        if key == 'distance':
            # round the distance to the nearest integer
            df['round_shot_distance'] = df['shot_distance'].round(0)
            # compute goal rate for each distance
            goal_rate = df.groupby(['round_shot_distance'])[
                'isgoal'].mean().reset_index()
            # plot the goal rate
            p = sns.lineplot(x='round_shot_distance',
                             y='isgoal', data=goal_rate)
            # set axis labels and title
            p.set_xlabel('Distance from net')
            p.set_ylabel('Goal rate')
            p.set_title('Goal rate by distance from net')

        elif key == 'angle':
            # round the angle to the nearest integer
            df['round_angle'] = df['angle'].round(0)
            # compute goal rate for each angle
            goal_rate = df.groupby(['round_angle'])[
                'isgoal'].mean().reset_index()
            # plot the goal rate
            p = sns.lineplot(x='round_angle', y='isgoal', data=goal_rate)
            # set axis labels and title
            p.set_xlabel('Angle from net')
            p.set_ylabel('Goal rate')
            p.set_title('Goal rate by angle from net')

        if save:
            plt.savefig(path)

        plt.show()

    def empty_net_histogram(self, path: str, save: bool = False):
        """
        Plots the histogram of goal counts (empty net and not empty net seperated), binned by distance

        Parameters
        ----------
        path: str
            The path to save the image
        save: bool
            Whether to save the image or not

        Returns
        -------
        None
        """
        # make a copy of the data
        df = copy.deepcopy(self.data)
        # filter the data to only include goals
        df = df[df['isgoal'] == 1]
        # plot histogram of goal counts based on the empty net binned by distance
        p = sns.displot(
            df, x=df['shot_distance'], hue=df['isEmptyNet'], bins=30, multiple="stack")
        # set axis labels and title
        p.set_axis_labels("Distance from net", "Number of goals")
        p.fig.suptitle(
            "Distribution of goals by distance from net and empty net")
        # set figure size
        p.fig.set_size_inches(10, 5)
        plt.show()

        if save:
            p.savefig(path)


if __name__ == "__main__":
    # initialize experiment on comet.ml to track experiments and results
    # exp = experiment_init()

    # initialize feature engineering class
    fe = feature_engineering()

    # read data from csv file into a dataframe and store it in the class attribute
    fe.read_data(os.environ.get('TIDY_DATA_PATH'))

    # compute distance from net and angle from net for each shot and store it in the dataframe as new columns
    fe.distance_from_net()
    fe.angle_from_net()

    # compute if the shot is a goal and store it in the dataframe as a new column
    fe.is_goal()

    # compute if the net is empty and store it in the dataframe as a new column
    fe.empty_net()

    # plot histogram of shot counts binned by distance
    fe.shot_counts_histogram(bin_type='distance', path=os.environ.get(
        'IMAGE_PATH') + 'distance.png', save=True)

    # plot histogram of shot counts binned by angle
    fe.shot_counts_histogram(
        bin_type='angle', path=os.environ.get(
            'IMAGE_PATH') + 'angle.png', save=True)

    # plot jointplot of angle vs distance from net
    # kind can be different types of plots (scatter, kde, hist, hex, reg, resid) default: scatter
    fe.angle_distance_joinplot(path=os.environ.get(
        'IMAGE_PATH') + 'angle_distance.png', save=True, kind='scatter')

    # plot goal rate by distance from net
    fe.plot_goal_rate(path=os.environ.get(
        'IMAGE_PATH') + 'goal_rate_distance.png', save=True, key='distance')

    # plot goal rate by angle from net
    fe.plot_goal_rate(path=os.environ.get(
        'IMAGE_PATH') + 'goal_rate_angle.png', save=True, key='angle')

    # plot histogram of goal counts based on the empty net binned by distance
    fe.empty_net_histogram(path=os.environ.get(
        'IMAGE_PATH') + 'empty_net.png', save=True)

    # save the data with the new features
    fe.save_data(os.environ.get('SAVE_DATA_PATH'))
