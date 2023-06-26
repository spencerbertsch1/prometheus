
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
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS, direction_dict
from wildfire_env_v3 import WildfireEnv
from utils import Cumulative


class Heuristic():

    def __init__(self, name: str, env, verbose: bool = True):
        # define the heuristic name
        self.name = name
        self.env = env
        self.verbose = verbose
        # the stage of the heuristic to be executed 
        self.stage = 1
        # the current list of actions to take in order to complete the current stage 
        self.actions_list = []

    def get_path_to_drop_site(self, obs: dict):
        # return random list as placeholder
        # return [self.env.action_space.sample() for x in range(20)]
        return [0 for x in range(10)]

    def get_path_to_closest_airport(self, obs: dict):
        # return random list as placeholder
        # return [self.env.action_space.sample() for x in range(20)]
        return [2 for x in range(10)]

    def predict(self, obs: dict):
        helicopter = obs['agent_list'][0]  # <-- later on we will iterate over all agents

        # STAGE 1: Aircraft refuals at the airport and takes off 
        if helicopter.location in obs['airport_locations']:
            self.stage = 1
            self.actions_list = self.get_path_to_drop_site(obs=obs)

        # STAGE 2: Aircraft drops fire retardant perpendicular to the wind direction 
        if ((self.stage == 1) and (len(self.actions_list) == 0)) or \
            ((self.stage == 2) and (helicopter.phoschek_level > 0)):
            self.stage == 2
            helicopter.dropping_phoschek = True

            # get perpendicular wind direction action
            wind_lst = list(direction_dict.values())
            wind_lst.append([0, 1])
            down_wind_action = direction_dict[EnvParams.wind]
            perp_wind_action = wind_lst[down_wind_action-2]

            return perp_wind_action

        # STAGE 3: move to the closest airport to refuel and get more phoschek
        if helicopter.phoschek_level == 0:
            self.actions_list = self.get_path_to_closest_airport()

        # TAKE ACTION
        # take an action along the path to the current goal for this stage 
        action = self.actions_list[0]
        self.actions_list = self.actions_list[1:]

        return action

        


