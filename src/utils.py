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
import seaborn as sns
import time 
from pathlib import Path
import sys
import matplotlib.animation as animation
from matplotlib.pyplot import imread
from PIL import Image, ImageDraw
import cv2

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
def animate(i, num, frames, patch, norm_alphas):
    if i%10==0: 
        LOGGER.info(f'Currently writing frame {i} of {len(frames)} to animation.')
    patch.set_data(np.fliplr(np.rot90(frames[i], 2)))
    patch.set_alpha(np.fliplr(np.rot90(norm_alphas[i], 2)))
    return patch,


# define helper function
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation_v2(frames, repeat: bool, interval: int, show_background_image: bool, 
                      save_anim: bool, show_anim: bool, alphas: np.array) -> animation.FuncAnimation:
    """
    Function to plot the series of environment frames generated during training 
    """
    # Colours for visualization: brown for EMPTY, dark green for TREE and orange
    # for FIRE. Note that for the colormap to work, this list and the bounds list
    # must be one larger than the number of different values in the array.
    # EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT = 0, 1, 2, 3, 4, 5

    if show_background_image: 
        img = imread("src/data/SF_map_very_small.jpeg")
        norm_alphas = []
        for i in range(len(frames)):
            arr = alphas[i].copy()
            arr[arr != 0] = 1
            norm_alphas.append(arr)

    if AnimationParams.show_full_anim: 
         pass
    else: 
        if EnvParams.fire_speed != 1:
            frames = frames[0::EnvParams.fire_speed]
            norm_alphas = norm_alphas[0::EnvParams.fire_speed]
    
    # choose the colors for each element of the image
    # https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    colors_list = ['black', 'forestgreen', 'orange', 'white', 'red', 'lightblue']
    cmap = colors.ListedColormap(colors_list)
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()

    # set the resolution for the animation
    fig.set_size_inches(EnvParams.grid_size/40, EnvParams.grid_size/40, True)
    patch = plt.imshow(frames[0], cmap=cmap, norm=norm, interpolation='nearest', zorder=3, alpha=norm_alphas[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, animate, fargs=(i, frames, patch, norm_alphas),
        frames=len(frames), repeat=repeat, interval=interval, blit=False)

    if show_background_image: 
        arr = frames[0]
        extent = [0, arr.shape[0]-1, 0, arr.shape[1]-1]
        plt.imshow(img, zorder=0,  extent=extent, alpha=1)
    
    if save_anim: 
        fps = int(round(1000/interval))
        anim_path: Path = ABSPATH_TO_ANIMATIONS / 'wildfire_episode.mp4'
        anim.save(anim_path, writer='ffmpeg', fps=fps)

    if show_anim: 
        plt.show()

    return anim


def plot_animation(frames, repeat: bool, interval: int, save_anim: bool, show_anim: bool) -> animation.FuncAnimation:
    """
    Function to plot the series of environment frames generated during training 
    """
    # Colours for visualization: brown for EMPTY, dark green for TREE and orange
    # for FIRE. Note that for the colormap to work, this list and the bounds list
    # must be one larger than the number of different values in the array.
    # EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT = 0, 1, 2, 3, 4, 5
    if AnimationParams.show_full_anim: 
         pass
    else: 
        if EnvParams.fire_speed != 1:
            frames = frames[0::EnvParams.fire_speed]
    
    # choose the colors for each element of the image
    # https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    colors_list = ['black', 'forestgreen', 'orange', 'white', 'red', 'lightblue']
    cmap = colors.ListedColormap(colors_list)
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    patch = plt.imshow(frames[0], cmap=cmap, norm=norm, interpolation='nearest')
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


def get_fire_centroid(env_state: np.array, verbose = False):
    """
    Simple utility function to get the centroid of the currently burning nodes.     
    """

    if verbose: 
        ax = sns.heatmap(env_state, linewidth=0)
        plt.show()

    # calculate the centroid of the currently burning nodes 
    count = (env_state == 1).sum()
    y_center, x_center = np.argwhere(env_state==1).sum(0)/count
    centroid = {'x_center': x_center, 'y_center': y_center}

    return centroid


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