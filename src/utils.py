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


def Cumulative(lst):
    """
    Simple utility function to return the cumulative list given a list of ints or floats
    """
    cu_list = []
    length = len(lst)
    cu_list = [sum(lst[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]


def numpy_element_counter(arr: np.array) -> dict:
    """
    Utility function to return a dict containing the counts for elements in np array
    """
    unique, counts = np.unique(arr, return_counts=True)
    counts_dict: dict = dict(zip(unique, counts))
    return counts_dict


# define helper function
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat: bool, interval: int, save_anim: bool, show_anim: bool) -> animation.FuncAnimation:
    """
    Function to plot the series of environment frames generated during training 
    """
    # Colours for visualization: brown for EMPTY, dark green for TREE and orange
    # for FIRE. Note that for the colormap to work, this list and the bounds list
    # must be one larger than the number of different values in the array.
    # EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT = 0, 1, 2, 3, 4, 5
    if EnvParams.fire_speed != 1:
         frames = frames[0::EnvParams.fire_speed]
    
    colors_list = ['black', 'forestgreen', 'orange', 'white', 'red', 'lightblue']
    cmap = colors.ListedColormap(colors_list)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    patch = plt.imshow(frames[0], cmap=cmap, interpolation='nearest')
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    
    if save_anim: 
        fps = int(round(1000/interval))
        anim_path: Path = ABSPATH_TO_ANIMATIONS / 'wildfire_episode.mp4'
        anim.save(anim_path, writer='ffmpeg', fps=fps)

    if show_anim: 
        plt.show()

    return anim

import pygame


def viewer(X: np.array):
    pygame.init()
    display = pygame.display.set_mode((750, 750))
    surf = pygame.surfarray.make_surface(X)

    pygame.display.set_caption('Wildfire Containment Episode')

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        display.blit(surf, (0, 0))
        pygame.display.update()
    pygame.quit()


class Aircraft():
    aircraft_type = 'Aircraft'
    def __init__(self, fuel_level: float = 1.0, phoschek_level: float = 1.0, curr_direction: str = 'N', 
                     dropping_phoschek: bool = False, location: list = [1, 1], curr_terrain = 0):
        self.fuel_level = fuel_level
        self.phoschek_level = phoschek_level
        self.curr_direction = curr_direction
        self.dropping_phoschek = dropping_phoschek
        self.location = location
        self.curr_terrain = curr_terrain

    def __repr__(self):
        print_str = f'''
                      {self.aircraft_type} fuel level: {round(self.fuel_level * 100, 2)}% 
                      {self.aircraft_type} phoschek level: {round(self.phoschek_level * 100, 2)}%
                      {self.aircraft_type} current direction: {self.curr_direction}
                      {self.aircraft_type} currently dropping phoschek: {self.dropping_phoschek}
                      {self.aircraft_type} location: {self.location}
                      '''
        print(print_str)
        
class Helicopter(Aircraft):
        aircraft_type = 'Helicopter'
        def __init__(self, fuel_level: float = 1.0, phoschek_level: float = 1.0, curr_direction: str = 'N', 
                     dropping_phoschek: bool = False, location = [1, 1]):
            super().__init__(fuel_level, phoschek_level, curr_direction, dropping_phoschek, location)


class Plane(Aircraft):
        aircraft_type = 'Plane'
        def __init__(self, fuel_level: float = 1.0, phoschek_level: float = 1.0, curr_direction: str = 'N', 
                     dropping_phoschek: bool = False, location = [1, 1]):
            super().__init__(fuel_level, phoschek_level, curr_direction, dropping_phoschek, location)