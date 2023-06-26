"""
-------------------------------
| Dartmouth College           |
| RL 4 Wildfire Containment   |
| 2023                        |
| Spencer Bertsch             |
-------------------------------

This script contains the functions needed to simulate the wildfire containment environment. 
"""


# imports 
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
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS, \
    EMPTY, TREE, FIRE, AIRCRAFT, PHOSCHEK, AIRPORT, direction_dict, action_to_direction
from utils import numpy_element_counter, plot_animation

# import environment, agent, and animation parameters 
WIND = EnvParams.wind
FOREST_FRACTION = EnvParams.forest_fraction
FIRE_SPREAD_PROB = EnvParams.fire_spread_prob
UP_WIND_SPREAD_PROB = EnvParams.up_wind_spread_prob
FIRE_SPEED = EnvParams.fire_speed
GRID_SIZE = EnvParams.grid_size
IGNITION_POINTS = EnvParams.ignition_points

# set the random seed for predictable runs 
np.random.seed(0)

# Colours for visualization: brown for EMPTY, dark green for TREE and orange
# for FIRE. Note that for the colormap to work, this list and the bounds list
# must be one larger than the number of different values in the array.
colors_list = ['black', 'green', 'darkorange', 'grey', 1]
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)


def iterate_fire_v2(X: np.array, phoschek_array: np.array, i: int):
    """
    Iterate the forest according to the forest-fire rules.
    https://scipython.com/blog/the-forest-fire-model/
    
    https://stackoverflow.com/questions/63661231/replacing-numpy-array-elements-that-are-non-zero
    ^ This greatly accelerated the run time of this function
    """
   
    if i % EnvParams.fire_speed == 0:

        # define environment parameters based on the current configuration
        ny, nx = EnvParams.grid_size, EnvParams.grid_size
        wind = EnvParams.wind
        fire_spread_prob = EnvParams.fire_spread_prob
        up_wind_spread_prob = EnvParams.up_wind_spread_prob
        
        # The boundary of the forest is always empty, so only consider cells
        # indexed from 1 to nx-2, 1 to ny-2

        X1 = np.zeros((ny, nx))
        nodes_searched = 0
        curr_burning_nodes = 0
        
        np.copyto(X1, X, where=X != EMPTY)

        # iterate over the currently burning nodes 
        for ix in range(1,nx-1):
            for iy in range(1,ny-1):
                if X[iy,ix] == FIRE:
                    curr_burning_nodes += 1
                    X1[iy,ix] = EMPTY
                    
                    # iterate over each of the 8 actions and their respective neighbor nodes
                    # here we index by dy then dx since that's how Numpy indexing works 
                    for action, (dy,dx) in action_to_direction.items():
                        # print(f'action: {action}, direction: {direction_dict[wind]}, dx: {dx}, dy: {dy}')
                        nodes_searched += 1
                        # The diagonally-adjacent trees are further away, so
                        # only catch fire with a reduced probability:
                        if abs(dx) == abs(dy) and wind == 'none' and np.random.random() < 0.573:
                            continue

                        # this prevents straight right corners from occuring in the fire frontier
                        # TODO fix fire from going out 9% of the time 
                        if abs(dx) == abs(dy) and wind != 'none' and np.random.random() < 0.08:
                            continue
                        
                        if X[iy+dy,ix+dx] == TREE:
                            # in this case there is no wind so the fire spreads radially outwards
                            if wind == 'none': 
                                if np.random.random() < fire_spread_prob:
                                    if ((phoschek_array[iy+dy,ix+dx] == 0) & (phoschek_array[iy+dy,ix] == 0) & (phoschek_array[iy,ix+dx] == 0)):
                                        X1[iy+dy,ix+dx] = FIRE
                                    # break
                            else:
                                # account for wind
                                if action==direction_dict[wind]: 
                                    if ((phoschek_array[iy+dy,ix+dx] == 0) & (phoschek_array[iy+dy,ix] == 0) & (phoschek_array[iy,ix+dx] == 0)):
                                        X1[iy+dy,ix+dx] = FIRE
                                
                                # add additional condition to slow fire spread based on probability of spreading
                                else:
                                    if np.random.random() < up_wind_spread_prob:
                                        if ((phoschek_array[iy+dy,ix+dx] == 0) & (phoschek_array[iy+dy,ix] == 0) & (phoschek_array[iy,ix+dx] == 0)):
                                            X1[iy+dy,ix+dx] = FIRE
                            
        np.copyto(X1, phoschek_array, where=phoschek_array != EMPTY)

        return {"X": X1, "nodes_processed": nodes_searched, 'curr_burning_nodes': curr_burning_nodes}
    
    # slow fire's progession
    else:
        return {"X": X, "nodes_processed": 0, 'curr_burning_nodes': 0}


def initialize_env() -> np.array:
    """
    Helper function that initializes the np array representing the starting environment state. 
    """
    # define the grid size 
    nx, ny = GRID_SIZE, GRID_SIZE
    # Initialize the forest grid.
    X  = np.zeros((ny, nx))
    X[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
    X[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < FOREST_FRACTION

    if IGNITION_POINTS == 1:
        # we only have one ignition point at the center of the environment
        ignition_index = int(GRID_SIZE/2)
        X[ignition_index, ignition_index] = FIRE
        X[ignition_index+1, ignition_index] = FIRE
        X[ignition_index, ignition_index+1] = FIRE
        X[ignition_index-1, ignition_index] = FIRE
        X[ignition_index, ignition_index-1] = FIRE
    else:
        # randomly select some ignition points
        point_list = [int(GRID_SIZE - GRID_SIZE/5), int(GRID_SIZE/2), int(GRID_SIZE/3), int(GRID_SIZE - GRID_SIZE/7), int(GRID_SIZE/9), int(GRID_SIZE/4)]
        for i in range(IGNITION_POINTS):
            X[np.random.choice(point_list), np.random.choice(point_list)] = FIRE

    # TODO SUNSET this old code - we now have a separate array for the airports
    # airport #1 
    # X[GRID_SIZE-3, GRID_SIZE-3] = AIRPORT
    # airport #2 
    # X[3, GRID_SIZE-3] = AIRPORT

    return X


def main():

    tic = time.time()

    X = initialize_env()

    phoschek_array = np.zeros((EnvParams.grid_size, EnvParams.grid_size))

    frames = []

    # run the RL algorithm and get the frames of the environment state 
    done = False
    for i, step in enumerate(range(10_000)):
        X_dict = iterate_fire_v2(X=X, phoschek_array=phoschek_array, i=i)
        X = X_dict['X']
        frames.append(X_dict['X'])

        # log the progress 
        if i%10==0:
            cnt = X_dict['nodes_processed']
            LOGGER.info(f'Nodes processed on step {i}: {cnt}')

        # test to see if there are any fire nodes remaining in the environment 
        node_cnt_dict = numpy_element_counter(arr=X)
        dict_keys = list(node_cnt_dict.keys())
        if FIRE not in dict_keys: 
            done = True

        if done:
            break

    toc = time.time()
    print(f'Episode took {round(toc-tic, 2)} seconds.')
    
    plot_animation(frames=frames, repeat=AnimationParams.repeat, interval=AnimationParams.interval, 
                   save_anim=AnimationParams.save_anim, show_anim=AnimationParams.show_anim)


if __name__ == "__main__":
    main()

