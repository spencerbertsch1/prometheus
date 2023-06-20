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
print(f'Working directory: {PATH_TO_WORKING_DIR}')
sys.path.append(str(PATH_TO_WORKING_DIR))
from settings import LOGGER, AnimationParams, EnvParams, ABSPATH_TO_ANIMATIONS
from utils import numpy_element_counter

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

# Displacements from a cell to its eight nearest neighbours
neighbourhood = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE, PLANE = 0, 1, 2, 3
wind_dict_v2 = {'N':3, 'NE':5, 'E':6, 'SE':7, 'S':4, 'SW':2, 'W':1, 'NW':0}
# Colours for visualization: brown for EMPTY, dark green for TREE and orange
# for FIRE. Note that for the colormap to work, this list and the bounds list
# must be one larger than the number of different values in the array.
colors_list = ['black', 'green', 'darkorange', 'grey']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)


def iterate_v2(X, nx, ny, fire_spread_prob: float, up_wind_spread_prob: float, wind: str):
    """Iterate the forest according to the forest-fire rules."""

    # The boundary of the forest is always empty, so only consider cells
    # indexed from 1 to nx-2, 1 to ny-2
    X1 = np.zeros((ny, nx))
    cnt = 0
    
    # fill in all the trees 
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            if X[iy,ix] == TREE:
                X1[iy,ix] = TREE
                cnt += 1

    # iterate over the currently burning nodes 
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            if X[iy,ix] == FIRE:
                X1[iy,ix] = EMPTY
                for i, (dx,dy) in enumerate(neighbourhood):
                    cnt += 1
                    # The diagonally-adjacent trees are further away, so
                    # only catch fire with a reduced probability:
                    if abs(dx) == abs(dy) and wind == 'none' and np.random.random() < 0.573:
                        continue

                    # this prevents straight right corners from occuring in the fire frontier
                    if abs(dx) == abs(dy) and wind != 'none' and np.random.random() < 0.07:
                        continue
                    
                    if X[iy+dy,ix+dx] == TREE:
                        # in this case there is no wind so the fire spreads radially outwards
                        if wind == 'none': 
                            if np.random.random() < fire_spread_prob:
                                X1[iy+dy,ix+dx] = FIRE
                                # break
                        else:
                            # account for wind
                            if i==wind_dict_v2[wind]: 
                                X1[iy+dy,ix+dx] = FIRE
                            
                            # add additional condition to slow fire spread based on probability of spreading
                            else:
                                if np.random.random() < up_wind_spread_prob:
                                    X1[iy+dy,ix+dx] = FIRE
                        
    return {"X": X1, "nodes_processed": cnt}


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
    else:
        # randomly select some ignition points
        point_list = [int(GRID_SIZE - GRID_SIZE/5), int(GRID_SIZE/2), int(GRID_SIZE/3), int(GRID_SIZE - GRID_SIZE/7), int(GRID_SIZE/9), int(GRID_SIZE/4)]
        for i in range(IGNITION_POINTS):
            X[np.random.choice(point_list), np.random.choice(point_list)] = FIRE

    return X


def main():

    tic = time.time()

    X = initialize_env()

    frames = []

    # run the RL algorithm and get the frames of the environment state 
    done = False
    for i, step in enumerate(range(10_000)):
        X_dict = iterate_v2(X, nx=GRID_SIZE, ny=GRID_SIZE, fire_spread_prob=FIRE_SPREAD_PROB, 
                    up_wind_spread_prob=UP_WIND_SPREAD_PROB, wind=WIND)
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

"""
TODOs

- add the rest of the OpenAI gym notation back into the main() function
- add 

- add single agent to the system
- add random action function (starting with moves)

"""