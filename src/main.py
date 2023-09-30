
"""
-------------------------------
| Dartmouth College           |
| RL 4 Wildfire Containment   |
| 2023                        |
| Spencer Bertsch             |
-------------------------------

This script contains utility functions used through out this project. 
"""

# imports 
import pygame
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import colors
import time 
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px

# local imports
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_WORKING_DIR: Path = PATH_TO_THIS_FILE.parent.parent
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS, \
    AgentParams, ABSPATH_TO_RESULTS
from wildfire_env_v3 import WildfireEnv
from utils import Cumulative, Helicopter, visualize_episodes

# import environment, agent, and animation parameters 
def run_episodes(num_episodes: int, verbose: bool = False):
    """
    Central function used to run some number of training episodes 
    """

    burn_lists = []

    for ep in range(num_episodes):

        # update the random seed so that each run is predictable, but different
        np.random.seed(ep+1)

        env = WildfireEnv(render_mode='rgb_array')
        obs, info = env.reset()

        i = 0
        curr_burning_nodes_lst = []
        while True:
            i += 1
            # Take a random action
            # action = env.action_space.sample()
            # action = 0
            action = env.heuristic2(obs=obs)
            obs, reward, done, info = env.step(action, i=i)
            
            # Render the game
            # env.render()

            # store environment info for plotting
            curr_burning_nodes = info['curr_burning_nodes']
            curr_burning_nodes_lst.append(curr_burning_nodes)
            
            if done == True:
                break

        env.close()

        c_burning_lst = Cumulative(lst=curr_burning_nodes_lst)
        burn_lists.append(c_burning_lst)

    if verbose: 
        visualize_episodes(burn_lists=burn_lists)

    burned_nodes_lst: list = [burn_lists[i][-1] for i in range(len(burn_lists))]
    return {'burned_nodes_lst': burned_nodes_lst}


def generate_results(save_results: bool = True):

    results_df = pd.DataFrame([])

    for fire_speed in [6, 7, 8, 9]:
        for drop_rate in [0.05,0.04, 0.03, 0.005]:

            # set global configs 
            AgentParams.phoscheck_drop_rate = drop_rate
            EnvParams.fire_speed = fire_speed

            # run episodes with new configs
            results_dict = run_episodes(num_episodes=15)
            num_burned_nodes: list = results_dict['burned_nodes_lst']

            # store the results in a dataframe for future use 
            col_name: str = f'capacity{round(1/AgentParams.phoscheck_drop_rate)}-speed{EnvParams.fire_speed}'
            results_df[col_name] = num_burned_nodes

            print(f'Number of nodes burned during each episode: {num_burned_nodes}')

    if save_results:
        path_to_results: Path = ABSPATH_TO_RESULTS / 'heuristics' / 'results.csv'
        results_df.to_csv(str(path_to_results), sep=',')

    return results_df

if __name__ == "__main__":
    # results_dict = run_episodes(num_episodes=1)
    df: pd.DataFrame = generate_results(save_results=True)
