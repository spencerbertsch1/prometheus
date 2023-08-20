"""
-------------------------------
| Dartmouth College           |
| RL 4 Wildfire Containment   |
| 2023                        |
| Spencer Bertsch             |
-------------------------------

This settings file contains basic utilities and information required by python scripts through out
this project. 

Tools such as the logger and information such as the dynamic path to the wildfire datasets 
are stored here and easily accessed from anywhere in the project. 
"""


import os
from pathlib import Path
import toml
import json
from typing import Dict, Any
import logging
import numpy as np

# define paths to files and outputs
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
ABSPATH_TO_TOML: Path = PATH_TO_THIS_FILE.parent / "config.toml"
ABSPATH_TO_ANIMATIONS: Path = PATH_TO_THIS_FILE.parent / "src" / "animations"
ABSPATH_TO_DATA: Path = PATH_TO_THIS_FILE.parent / "src" / "data" 

EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT = 0, 1, 2, 3, 4, 5

direction_dict = {'N':0, 'NE':1, 'E':2, 'SE':3, 'S':4, 'SW':5, 'W':6, 'NW':7}
"""
The following dictionary maps abstract actions from `self.action_space` to
the direction we will walk in if that action is taken.
For example 0 is North, 1 is NE, etc. See above dictionary for mapping. 
"""
action_to_direction = {
            0: np.array([-1, 0]),   # N
            1: np.array([-1, 1]),   # NE
            2: np.array([0, 1]),    # E 
            3: np.array([1, 1]),    # SE
            4: np.array([1, 0]),    # S
            5: np.array([1, -1]),   # SW
            6: np.array([0, -1]),   # W
            7: np.array([-1, -1]),  # NW
        }
direction_to_action = {tuple(v): k for k, v in action_to_direction.items()}

# use toml.load to read `config.toml` file in as a dictionary
CONFIG_DICT: Dict[Any, Any] = toml.load(str(ABSPATH_TO_TOML))

class EnvParams:
    wind: bool = CONFIG_DICT['env']['wind']
    forest_fraction: bool = CONFIG_DICT['env']['forest_fraction']
    fire_speed: bool = CONFIG_DICT['env']['fire_speed']
    grid_size: bool = CONFIG_DICT['env']['grid_size']
    ignition_points: bool = CONFIG_DICT['env']['ignition_points']
    alpha: bool = CONFIG_DICT['env']['alpha']
    w: bool = CONFIG_DICT['env']['w']
    epsilon: bool = CONFIG_DICT['env']['epsilon']
    delt: bool = CONFIG_DICT['env']['delt']

class AgentParams:
    phoscheck_drop_rate: bool = CONFIG_DICT['agent']['phoscheck_drop_rate']

class AnimationParams:
    repeat: bool = CONFIG_DICT['animation']['repeat']  # repeat the animation
    interval: bool = CONFIG_DICT['animation']['interval'] # millisecond interval for each frame in the animation
    save_anim: bool = CONFIG_DICT['animation']['save_anim']  # save the animation to disk 
    show_anim: bool = CONFIG_DICT['animation']['show_anim']  # show animation after each episode
    show_full_anim: bool = CONFIG_DICT['animation']['show_full_anim']  # show full animation (frame by frame)
    dev_fast_anim: bool = CONFIG_DICT['animation']['dev_fast_anim']  # greatly speed up animation (for development)
    show_background_image: bool = CONFIG_DICT['animation']['show_background_image']  # show landscape image



def create_logger(logger_name: str) -> logging.Logger:
    """
    Function to create a generic logger which can be used through out different scripts in the migraine
    predictor project
    :arg: logger_name, str - the name of the specific logger which should be used. DATA_CLEAN for example
    """
    # define the 'top_level" name, or the name of the entire project that will produce logs
    top_level_name: str = 'PyFire-Dev'

    # create logger
    logger = logging.getLogger(top_level_name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    path_to_log_file: Path = PATH_TO_THIS_FILE.parent / f'{top_level_name}.log'
    # set the new location as the directory/file location for the logging output
    fh = logging.FileHandler(path_to_log_file)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(f'{logger_name} - %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# create the logger that will be used through out this project
LOGGER = create_logger(logger_name="Wildfire Dev")