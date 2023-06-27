
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

# local imports
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_WORKING_DIR: Path = PATH_TO_THIS_FILE.parent.parent
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS
from wildfire_env_v3 import WildfireEnv
from utils import Cumulative
from heuristic import Heuristic


# import environment, agent, and animation parameters 
def main():
    env = WildfireEnv(render_mode='rgb_array')
    obs, info = env.reset()

    heuristic = Heuristic(name='basic_heuristic', env=env, verbose=True)

    i = 0
    curr_burning_nodes_lst = []
    while True:
        i += 1
        # Take a random action
        # action = env.action_space.sample()
        action = heuristic.predict(obs=obs)
        obs, reward, done, info = env.step(action, i=i)
        
        # Render the game
        env.render()

        # store environment info for plotting
        curr_burning_nodes = info['curr_burning_nodes']
        curr_burning_nodes_lst.append(curr_burning_nodes)
        
        if done == True:
            break

    env.close()

    c_burning_lst = Cumulative(lst=curr_burning_nodes_lst)

    # df = pd.DataFrame({'x_data':range(len(c_burning_lst)), 'y_data':c_burning_lst})
    # fig = px.line(df, x='x_data', y='y_data', title="Testing")
    # fig.show()

if __name__ == "__main__":
    main()
