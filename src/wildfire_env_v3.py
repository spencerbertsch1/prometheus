import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors

from pathlib import Path
import sys

# local imports
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_WORKING_DIR: Path = PATH_TO_THIS_FILE.parent.parent
print(f'Working directory: {PATH_TO_WORKING_DIR}')
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS
from fire_sim_v2 import iterate_fire_v2, initialize_env
from utils import numpy_element_counter, plot_animation, viewer

# set the random seed for predictable runs 
SEED = 0
np.random.seed(SEED)

# Displacements from a cell to its eight nearest neighbours
neighbourhood = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE, PLANE, AIRPORT = 0, 1, 2, 3, 4
wind_dict_v2 = {'N':3, 'NE':5, 'E':6, 'SE':7, 'S':4, 'SW':2, 'W':1, 'NW':0}


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode='human', size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.phoschek_array = np.zeros((size, size))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(9)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # we use the 8 neighbor model 
        self._action_to_direction = {
            0: np.array([0,-1]),   # N
            1: np.array([1,-1]),   # NE
            2: np.array([1, 0]),   # E 
            3: np.array([1, 1]),   # SE
            4: np.array([0, 1]),   # S
            5: np.array([-1,1]),   # SW
            6: np.array([-1,0]),   # W
            7: np.array([-1,-1]),  # NW
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # define a private method that translates an environment state into an observation
    # this method is not strictly necessary, but it's very helpful 
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    # define private method that returns auxillary information for the step() and reset() methods
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):

        # move the agent 
        if action <= 7:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        # drop phoschek 
        else:
            # We use `np.clip` to make sure we don't leave the grid
            print('something')
            self.phoschek_array[self._agent_location[1], self._agent_location[0]] = 1
            print(self.phoschek_array)

        # An episode is done iff the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


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

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # we use the 8 neighbor model 
        self._action_to_direction = {
            0: np.array([0,-1]),   # N
            1: np.array([1,-1]),   # NE
            2: np.array([1, 0]),   # E 
            3: np.array([1, 1]),   # SE
            4: np.array([0, 1]),   # S
            5: np.array([-1,1]),   # SW
            6: np.array([-1,0]),   # W
            7: np.array([-1,-1]),  # NW
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _render_frame(self, X: np.array):
        pygame_viewer = viewer(self._env_state)

    def step(self, action):

        # move the agent 
        if action <= 7:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, EnvParams.grid_size - 1
            )

        # drop phoschek 
        else:
            # We use `np.clip` to make sure we don't leave the grid
            print('Dropping Phos Chek')
            self.phoschek_array[self._agent_location[1], self._agent_location[0]] = 1
            print(self.phoschek_array)

        # Execute one time step in the environment
        X_dict = iterate_fire_v2(self._env_state)
        self._env_state = X_dict['X']
        self._env_state[self._agent_location[0], self._agent_location[1]] = PLANE
        self.frames.append(self._env_state)

        # test to see if there are any fire nodes remaining in the environment 
        done = False
        node_cnt_dict = numpy_element_counter(arr=self._env_state)
        dict_keys = list(node_cnt_dict.keys())
        if FIRE not in dict_keys: 
            done = True
            plot_animation(frames=self.frames, repeat=AnimationParams.repeat, interval=AnimationParams.interval, 
                save_anim=AnimationParams.save_anim, show_anim=AnimationParams.show_anim)

        # TODO get info from the iterate function - how many currently burning nodes, etc
        reward = 1
        info = {'info': 1}

        return self._env_state, reward, done, info

    def reset(self):
        # Reset the state of the environment
        # We need the following line to seed self.np_random
        super().reset(seed=SEED)

        self._env_state = initialize_env()

        # Choose the agent's starting location as the airport
        self._agent_location = [EnvParams.grid_size-3, EnvParams.grid_size-3]
        self._env_state[self._agent_location[0], self._agent_location[1]] = PLANE

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

    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break

    env.close()


if __name__ == "__main__":
    main()
