
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
import math
from pathlib import Path
import sys
from random import choices
import networkx as nx
import copy

# local imports
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_WORKING_DIR: Path = PATH_TO_THIS_FILE.parent.parent
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AgentParams, AnimationParams, EnvParams, \
     ABSPATH_TO_ANIMATIONS, direction_dict, direction_to_action
from wildfire_env_v3 import WildfireEnv
from utils import Cumulative, get_fire_centroid, get_path_to_point

def heuristic2(obs: dict, env: WildfireEnv):

    # define observation information 
    agent_list = obs['agent_list']
    helicopter1 = agent_list[0]
    agent_location = helicopter1.location
    airport_locations = obs['airport_locations']
    env_state = obs['env_state']
    phoschek_array = obs['phoschek_array']

    # if the agent is at an airport 
    if agent_location in airport_locations:
        c_dict: dict = get_fire_centroid(env_state=env_state, verbose=True)
        x_center, y_center = round(c_dict['x_center'], 1), round(c_dict['y_center'], 1)

        print(f'X Center: {x_center}, Y Center: {y_center}')

        path_list = get_path_to_point(start=agent_location, end=[x_center, y_center])

    if len(path_list) == 0:
        action = env.action_space.sample()

    h_dict = {
        'phase': 1, 
        'curr_path': path_list
    }

    return action, h_dict

