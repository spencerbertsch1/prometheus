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
from settings import LOGGER, AnimationParams, EnvParams, AgentParams, ABSPATH_TO_ANIMATIONS, \
    EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT, direction_dict, action_to_direction
from fire_sim_v2 import iterate_fire_v2, initialize_env
from utils import numpy_element_counter, plot_animation, viewer, Helicopter, Cumulative

# set the random seed for predictable runs 
SEED = 0
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
        self.phoschek_array = np.zeros((EnvParams.grid_size, EnvParams.grid_size))
        self.agent_array = np.zeros((EnvParams.grid_size, EnvParams.grid_size))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _render_frame(self, X: np.array):
        pygame_viewer = viewer(self._env_state)

    def step(self, action, i: int):

        # move the agent 
        if action <= 7:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 1, EnvParams.grid_size - 2
            )

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
        X_dict = iterate_fire_v2(X=self._env_state, phoschek_array=self.phoschek_array, i=i)
        self._env_state = X_dict['X']
        
        full_frame = self._env_state.copy()
        full_frame[self._agent_location[0], self._agent_location[1]] = AIRCRAFT
        
        self.frames.append(full_frame)

        # get counts from the environment
        node_cnt_dict = numpy_element_counter(arr=self._env_state)
        dict_keys = list(node_cnt_dict.keys())

        # test to see if there are any fire nodes remaining in the environment 
        done = False
        if FIRE not in dict_keys: 
            done = True
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
            LOGGER.info(f'Nodes processed on step {i}: {cnt}')

        return self._env_state, reward, done, info

    def reset(self):
        # Reset the state of the environment
        # We need the following line to seed self.np_random
        super().reset(seed=SEED)

        self._env_state = initialize_env()

        self.helicopter = Helicopter(location=[EnvParams.grid_size-10, EnvParams.grid_size-10])

        # Choose the agent's starting location as the airport
        self._agent_location = self.helicopter.location
        # self._env_state[self._agent_location[0], self._agent_location[1]] = AIRCRAFT

        observation = self._env_state
        info = {'info': 1}

        self.render()

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


# import environment, agent, and animation parameters 
def main():
    env = WildfireEnv(render_mode='rgb_array')
    obs = env.reset()

    i = 0
    curr_burning_nodes_lst = []
    while True:
        i += 1
        # Take a random action
        action = env.action_space.sample()
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


"""
TODOs 
2. Implement heuristic where the plane flies perpendicular to the wind and drops phos chek 
4. write outer loop function that tests different heuristics against one another and plots the cumulative area saved
3. embed this in a streamlit app
"""