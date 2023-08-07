import numpy as np
import pandas as pd
import pygame


import gymnasium as gym
from gymnasium import spaces

from matplotlib import pyplot as plt
import plotly.express as px
from matplotlib import animation
from matplotlib import colors

from pathlib import Path
import sys

# local imports
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_WORKING_DIR: Path = PATH_TO_THIS_FILE.parent.parent
print(f'Working directory: {PATH_TO_WORKING_DIR}')
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AnimationParams, EnvParams, AgentParams, ABSPATH_TO_ANIMATIONS, ABSPATH_TO_DATA, \
    EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT, direction_dict, action_to_direction
from fire_sim_v3 import iterate_fire_v3, initialize_env
from fire_sim_v4 import iterate_fire_v4, initialize_env
from utils import numpy_element_counter, plot_animation, viewer, Helicopter, Cumulative, \
    plot_animation_v2, get_path_to_point, get_fire_centroid, get_closest_airport

# set the random seed for predictable runs 
SEED = 1
np.random.seed(SEED)


class WildfireEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode):
        super().__init__()
        # The action and observation spaces need to be gym.spaces objects:
        self.action_space = spaces.Discrete(9)  # 8-neighbor model + phos_chek
        # Here's an observation space for 200 wide x 100 high RGB image inputs:
        self.observation_space = spaces.Box(low=0, high=5, shape=(EnvParams.grid_size, EnvParams.grid_size, 1), dtype=np.uint8)
        # define an empty list that will store all of the environmetn states (for episode animations)
        self.frames = []
        self.phoschek_array = self.generate_starting_phoschek_array()
        # store the alphas used for creating the animation later on
        self.alphas_list = []
        # define airport locations
        self._airport_locations = [[EnvParams.grid_size-3, EnvParams.grid_size-3], 
                                   [EnvParams.grid_size-3, 3]]
        # initialize action list to empty
        self._action_list = []
        # ensure the render mode is in the list of possible values
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _render_frame(self, X: np.array):
        pygame_viewer = viewer(self._env_state)

    def generate_starting_phoschek_array(self):
        """
        Simple utility function to initialize the 2D numpy array of phos chek 
        """
        if AnimationParams.show_background_image:
            # path: Path = ABSPATH_TO_DATA / "starting_phoschek_array.npy"
            # arr = np.load(str(path))
            # # replace all non-empty values with PHOSCHEK value
            # arr[arr > 100] = PHOSCHEK
            # print(np.amax(arr))
            arr = np.zeros((EnvParams.grid_size, EnvParams.grid_size))
        else:
            arr = np.zeros((EnvParams.grid_size, EnvParams.grid_size))

        return arr

    def step(self, action, i: int):

        # move the agent 
        if action <= 7:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 1, EnvParams.grid_size - 2
            )
            self.helicopter.location = list(np.clip(
                self._agent_location + direction, 1, EnvParams.grid_size - 2
            ))

        # initiate phoschek drop
        if ((action == 8) & (not self.helicopter.dropping_phoschek)):
            # start dropping phos chek
            self.helicopter.dropping_phoschek = True
            # Here we move the aircraft forward in the same direction of movement 
            direction = action_to_direction[direction_dict[self.helicopter.curr_direction]]
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, EnvParams.grid_size - 1
            )

        # drop phos chek if there is some left in the aircraft 
        if ((self.helicopter.phoschek_level > 0) & (self.helicopter.dropping_phoschek)): 
            self.phoschek_array[self._agent_location[0], self._agent_location[1]] = PHOSCHEK
            self.helicopter.phoschek_level -= AgentParams.phoscheck_drop_rate

        # Execute one time step in the environment
        X_dict = iterate_fire_v4(X=self._env_state, phoschek_array=self.phoschek_array, i=i)
        self._env_state = X_dict['X']
        
        full_frame = self._env_state.copy()
        full_frame[self._agent_location[0], self._agent_location[1]] = AIRCRAFT

        alpha_array = (self._airport_array + self.phoschek_array + full_frame) - 1
        self.alphas_list.append(alpha_array)
        
        self.frames.append(full_frame)

        # get counts from the environment
        node_cnt_dict = numpy_element_counter(arr=self._env_state)
        dict_keys = list(node_cnt_dict.keys())

        # test to see if there are any fire nodes remaining in the environment 
        done = False
        if FIRE not in dict_keys: 
            done = True
            if AnimationParams.show_background_image:
                plot_animation_v2(frames=self.frames, repeat=AnimationParams.repeat, interval=AnimationParams.interval, 
                    save_anim=AnimationParams.save_anim, show_anim=AnimationParams.show_anim, 
                    show_background_image=AnimationParams.show_background_image, alphas=self.alphas_list)
            else:
                plot_animation(frames=self.frames, repeat=AnimationParams.repeat, interval=AnimationParams.interval, 
                    save_anim=AnimationParams.save_anim, show_anim=AnimationParams.show_anim)
            
            info = {'curr_burning_nodes': 0}
        
        else:
            curr_burning_nodes: int = node_cnt_dict[FIRE]
            info = {'curr_burning_nodes': curr_burning_nodes}

        # TODO get info from the iterate function - how many currently burning nodes, etc
        reward = 1

        # log the progress 
        if i%10==0:
            cnt = X_dict['nodes_processed']
            LOGGER.info(f'Nodes processed on step {i}: {cnt}, Phoschek Level: {round(self.helicopter.phoschek_level, 2)}')

        observation = {'agent_list': [self.helicopter],
                       'airport_locations': self._airport_locations, 
                       'env_state': self._env_state, 
                       'phoschek_array': self.phoschek_array}

        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment
        # We need the following line to seed self.np_random
        super().reset(seed=SEED)

        self._env_state = initialize_env()

        # define the agent 
        self.helicopter = Helicopter(location=self._airport_locations[1])
        
        # generate a numpy array containing airport locations (used for animation)
        self._airport_array = np.zeros((EnvParams.grid_size, EnvParams.grid_size))
        
        # add the airport locations to the environment state
        for loc in self._airport_locations:
            self._env_state[loc[0], loc[1]] = AIRPORT
            self._airport_array[loc[0], loc[1]] = AIRPORT

        # set the initial location to the airport 
        self._agent_location = self.helicopter.location
        self._env_state[self._agent_location[0], self._agent_location[1]] = AIRCRAFT

        full_frame = self._env_state.copy()
        full_frame[self._agent_location[0], self._agent_location[1]] = AIRCRAFT
        
        self.frames.append(full_frame)

        self.alphas_list.append(self._airport_array.copy())

        observation = {'agent_list': [self.helicopter],
                       'airport_locations': self._airport_locations, 
                       'env_state': self._env_state, 
                       'phoschek_array': self.phoschek_array}
        info = {'info': 1}

        self.render()

        s = 10*'-'
        LOGGER.info(f'{s} Environment Reset Complete {s}')
        LOGGER.info(f'Airport Locations: {self._airport_locations}')
        LOGGER.info(f'Starting Agent Locations: {self._agent_location}')

        return observation, info

    def render(self, close=False):
        if self.render_mode == 'human':
            # render to screen
            # self._render_frame(X=self._env_state)
            pass  # TODO
        elif self.render_mode == 'rgb_array':
            # render to a NumPy 100x200x1 array
            # return self._env_state
            pass # TODO
        else:
            pass
            # raise an error, unsupported mode


    def heuristic2(self, obs: dict):

        # if the agent is at an airport 
        if list(self._agent_location) in self._airport_locations:
            self.helicopter.phoschek_level = 1.0
            self.helicopter.fuel_level = 1.0
            self.helicopter.dropping_phoschek = False
            c_dict: dict = get_fire_centroid(env_state=self._env_state, verbose=True)
            y_center, x_center = round(c_dict['x_center'], 1), round(c_dict['y_center'], 1)
            target = [y_center - 45, x_center]

            print(f'X Center: {x_center}, Y Center: {y_center}')

            self._action_list = get_path_to_point(start=self._agent_location, goal=target)

        if ((len(self._action_list) == 0) & (self.helicopter.phoschek_level > 0)):
            if not self.helicopter.dropping_phoschek:
                # initiate the phos chek drop if it hasn't happened already
                return 8
            else:
                return self.action_space.sample()

        if len(self._action_list) == 0:
            # action = self.action_space.sample()
            # get the position of the closest airport 
            closest_airport_location: list = get_closest_airport(agent_location=self._agent_location, airport_locations=self._airport_locations)
            # get the action path that leads to that airport 
            self._action_list = get_path_to_point(start=self._agent_location, goal=closest_airport_location)
        # else:
        action = self._action_list[0]
        self._action_list = self._action_list[1:]

        return action


"""
TODOs 
1. remove _agent_location and replace with the helicopter locaiton instance variable 
2. Refine heuristic where the plane flies perpendicular to the wind and drops phos chek 
4. write outer loop function that tests different heuristics against one another and plots the cumulative area saved
3. embed this in a streamlit app

6. Create realistic simulations using wildfire data

"""
