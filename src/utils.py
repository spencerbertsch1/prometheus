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
import pandas as pd
from matplotlib import animation
from matplotlib import colors
import plotly.express as px
import time 
from pathlib import Path
import sys
import matplotlib.animation as animation
from matplotlib.pyplot import imread
from PIL import Image, ImageDraw
import cv2
import random
import math

# local imports
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_WORKING_DIR: Path = PATH_TO_THIS_FILE.parent.parent
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS, \
     EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT, direction_to_action, action_to_direction


def get_closest_airport(agent_location: list, airport_locations: list):
    closest_airport_index: int = 0
    min_dist: float = math.dist(airport_locations[0], agent_location)
    for i, airport_location in enumerate(airport_locations):
        curr_dist: float = math.dist(airport_location, agent_location)
        if curr_dist < min_dist:
            closest_airport_index = i
            min_dist = curr_dist

    closest_airport_location: list = airport_locations[closest_airport_index]
    return closest_airport_location


def get_path_to_point(start: list[int, int], goal: list[int, int], verbose: bool = False) -> list:
    """
    Very important utility function! This function is used to generate the path of actions to take in order for
    and agent to reach a goal state given a current state. 

    :param: start - 2-length list describing the x and y coordinates of the starting state
    :param: goal - 2-length list describing the x and y coordinates of the goal state
    """
    goal[0] = int(round(goal[0]))
    goal[1] = int(round(goal[1]))
    print(f'GETTING PATH: Start: {start}, Goal: {goal}')
    
    temp = start.copy()
    path: list = [temp.copy()]

    while list(temp) != list(goal):

        if temp[0] < goal[0]:
            temp[0] += 1
        elif temp[0] > goal[0]:
            temp[0] -= 1

        if temp[1] < goal[1]:
            temp[1] += 1
        elif temp[1] > goal[1]:
            temp[1] -= 1 

        path.append(temp.copy())

    # if verbose: 
    #     print(path)
    #     # visualize path 
    #     A = np.zeros([100, 100])
    #     for loc in path: 
    #         A[loc[0]][loc[1]] = 1
    #     A[start[0]][start[1]] = 2
    #     A[goal[0]][goal[1]] = 2
    #     ax = sns.heatmap(A, linewidth=0, cmap="YlGnBu")
    #     plt.show()

    # get the list of actions that correspond to this path 
    actions_list = []
    for i in range(len(path)-1):
        pos1 = path[i]
        pos2 = path[i+1]

        dir = [pos2[0]-pos1[0], pos2[1]-pos1[1]]

        action = direction_to_action[tuple(dir)]
        actions_list.append(action)

    # if verbose: 
    #     print(actions_list)
    #     # visualize path again
    #     current_pos = start.copy()
    #     B = np.zeros([100, 100])
    #     for action in actions_list: 
    #         dir = action_to_direction[action]
    #         current_pos[0] += dir[0]
    #         current_pos[1] += dir[1]
    #         B[current_pos[0]][current_pos[1]] = 1
    #     B[start[0]][start[1]] = 2
    #     B[goal[0]][goal[1]] = 2
    #     ax = sns.heatmap(B, linewidth=0)
    #     plt.show()

    # we shuffle the actions to prevent slight corners from appearing in the route
    random.shuffle(actions_list)

    return actions_list


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

    # if verbose: 
    #     ax = sns.heatmap(env_state, linewidth=0)
    #     plt.show()

    # calculate the centroid of the currently burning nodes 
    count = (env_state == FIRE).sum()
    y_center, x_center = np.argwhere(env_state == FIRE).sum(0)/count
    centroid = {'y_center': y_center, 'x_center': x_center}

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


def visualize_episodes(num_episodes: int, burn_lists: list):
    """
    Function to generate a multi-line plot in plotly express showing the cumulative nodes burned for each 
    episode in the training run. 
    """
    longest_burn_list = 0
    for i in range(num_episodes):
        if len(burn_lists[i]) > longest_burn_list:
            longest_burn_list = len(burn_lists[i])

    for i in range(num_episodes):
        if len(burn_lists[i]) < longest_burn_list:
            # pad the current burn list with final cumulative score 
            num_missing_vals = longest_burn_list - len(burn_lists[i])
            for j in range(num_missing_vals):
                burn_lists[i].append(burn_lists[i][-1])

    # generate dataframe for plotting
    df = pd.DataFrame({'Timesteps':range(len(burn_lists[0])), 'ep-0':burn_lists[0]})
    for i in range(1, num_episodes, 1):
        df[f'ep-{i}'] = burn_lists[i]

    # plot the resulting data
    fig = px.line(df, x='Timesteps', y=df.columns[1:(num_episodes+1)], title="Number of Burned Nodes Over Each Episode")
    fig.show()



if __name__ == "__main__":

    # some driver code for the get_path_to_point function
    start = [np.random.choice(list(range(100))), np.random.choice(list(range(100)))]
    goal = [np.random.choice(list(range(100))), np.random.choice(list(range(100)))]
    get_path_to_point(start=start, goal=goal)
