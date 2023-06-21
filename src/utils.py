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
    fig = plt.figure()
    patch = plt.imshow(frames[0])
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
