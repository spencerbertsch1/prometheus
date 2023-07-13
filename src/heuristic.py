
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
from utils import Cumulative


class Heuristic():

    def __init__(self, name: str, env, verbose: bool, obs: dict):
        # define the heuristic name
        self.name = name
        self.env = env
        self.verbose = verbose
        # the stage of the heuristic to be executed 
        self.stage = 1
        # the current list of actions to take in order to complete the current stage 
        self.actions_list = []
        # define the list of agents
        self.helicopter = obs['agent_list'][0]

    def get_phoschek_path(self, obs: dict):
        path_len = int(1 / AgentParams.phoscheck_drop_rate) + 1
        # drop fire retardant randomly until its gone
        return choices(list(direction_dict.values()), k=path_len)
    
    def get_perp_phoschek_path(self, obs: dict):
        self.helicopter = obs['agent_list'][0]  # <-- later on we will iterate over all agents

        # STAGE 2: Aircraft drops fire retardant perpendicular to the wind direction 
        if ((self.stage == 1) and (len(self.actions_list) == 0)) or \
            ((self.stage == 2) and (self.helicopter.phoschek_level > 0)):
            self.stage == 2
            self.helicopter.dropping_phoschek = True

            # get perpendicular wind direction action
            wind_lst = list(direction_dict.values())
            wind_lst.append([0, 1])
            down_wind_action = direction_dict[EnvParams.wind]
            perp_wind_action = wind_lst[down_wind_action-2]

            return perp_wind_action

    def get_path_to_drop_site(self, obs: dict):
        # return random list as placeholder
        # return [self.env.action_space.sample() for x in range(20)]
        return [1 for x in range(25)]

    def get_path_to_closest_airport(self, obs: dict, agent):
        """
        Function to return a list of integers in the range 0-7 representing the actions to 
        take in order to travel to the nearest airport. 
        """

        # step 1: locate closest airport 
        airport_locations = obs['airport_locations']
        agent_location = agent.location
        closest_airport_index: int = 0
        min_dist: float = math.dist(airport_locations[0], agent_location)
        for i, airport_location in enumerate(airport_locations):
            curr_dist: float = math.dist(airport_location, agent_location)
            if curr_dist < min_dist:
                closest_airport_index = i
                min_dist = curr_dist

        closest_airport_location: list = airport_locations[closest_airport_index]

        # step 2: get the path to the closest airport 
        agent_loc = agent_location.copy()
        path_list = [agent_loc.copy()]
        while agent_loc != closest_airport_location:
            
            if agent_loc[0] < closest_airport_location[0]:
                agent_loc[0] += 1
            elif agent_loc[0] > closest_airport_location[0]:
                agent_loc[0] -= 1

            if agent_loc[1] < closest_airport_location[1]:
                agent_loc[1] += 1
            elif agent_loc[1] > closest_airport_location[1]:
                agent_loc[1] -= 1

            path_list.append(agent_loc.copy())

        # step 3: convert the path to a list of actions
        action_list = []
        for i in range(len(path_list)-1):
            pos1 = path_list[i]
            pos2 = path_list[i+1]

            direction = [0, 0]
            direction[0] = pos2[0] - pos1[0]
            direction[1] = pos2[1] - pos1[1]

            action = direction_to_action[tuple(direction)]
            action_list.append(action)

        action_list.append(action_list[-1])
            
        return action_list



    def take_action(self):
        # TAKE ACTION
        # take an action along the path to the current goal for this stage 
        action = self.actions_list[0]
        self.actions_list = self.actions_list[1:]

        return action


    def get_action(self, obs: dict):

        # self.helicopter = obs['agent_list'][0]  # <-- later on we will iterate over all agents
        # STAGE 1: Aircraft refuels at the airport and takes off 
        if self.helicopter.location in obs['airport_locations']:
            self.helicopter.phoschek_level = 1.0
            self.helicopter.dropping_phoschek = False
            self.stage = 1
            self.actions_list = self.get_path_to_drop_site(obs=obs)
            print(f'STAGE 1, Agent Location: {self.helicopter.location}')

        if len(self.actions_list) > 0: 
            action = self.take_action()
            return action
        else:

            # STAGE 1: Aircraft refuels at the airport and takes off 
            if self.helicopter.location in obs['airport_locations']:
                self.helicopter.phoschek_level = 1.0
                self.helicopter.dropping_phoschek = False
                self.stage = 1
                self.actions_list = self.get_path_to_drop_site(obs=obs)
                print(f'STAGE 1, Agent Location: {self.helicopter.location}')
            
            # STAGE 2: drop the fire retardant 
            elif ((self.helicopter.phoschek_level == 1.0) & (self.helicopter.location not in obs['airport_locations'])):
                self.helicopter.dropping_phoschek = True
                self.actions_list = self.get_phoschek_path(obs=obs)
                print(f'STAGE 2, Agent Location: {self.helicopter.location}')

            # STAGE 3: move to the closest airport to refuel and get more phoschek
            elif round(self.helicopter.phoschek_level, 2) == 0:
                self.actions_list = self.get_path_to_closest_airport(obs=obs, agent=self.helicopter)
                print(f'STAGE 3, Agent Location: {self.helicopter.location}')

            action = self.take_action()
            return action
            
"""
TODO 

clean up bugs
"""