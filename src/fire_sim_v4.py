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
import seaborn as sns
import numpy as np
from matplotlib import animation
from matplotlib import colors
import time 
from pathlib import Path
import sys
import math

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
ALPHA = EnvParams.fire_spread_prob
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

def get_num_burning_neighbors(iy: int, ix: int, X: np.array):

    # ix and iy are passed as 1-indexed values, but we need them to be 0-indexed values
    # ix -= 1
    # iy -= 1

    # upper left corner 
    if ((iy==1) & (ix == 1)):
        N = [X[iy+1, ix], X[iy, ix+1], X[iy+1, ix+1]]

    # upper right corner 
    elif ((iy==1) & (ix == GRID_SIZE-1)):
        N = [X[iy+1, ix], X[iy, ix-1], X[iy+1, ix-1]]

    # lower left corner 
    elif ((iy==GRID_SIZE-1) & (ix == 1)):
        N = [X[iy-1, ix], X[iy, ix+1], X[iy-1, ix+1]]
    
    # lower right corner 
    elif ((iy==GRID_SIZE-1) & (ix == GRID_SIZE-1)):
        N = [X[iy-1, ix], X[iy, ix-1], X[iy-1, ix-1]]

    # middle upper boundary
    elif ((iy == 1) & ((ix > 1) & (ix < GRID_SIZE-1))):
        N = [X[iy, ix-1], X[iy+1, ix-1], X[iy+1, ix], X[iy+1, ix+1], X[iy, ix+1]]

    # middle right boundary
    elif (((iy > 1) & (iy < GRID_SIZE-1)) & (ix == GRID_SIZE-1)):
        N = [X[iy-1, ix], X[iy-1, ix-1], X[iy, ix-1], X[iy+1, ix-1], X[iy+1, ix]]

    # middle bottom boundary
    elif ((iy == GRID_SIZE-1) & ((ix > 0) & (ix < GRID_SIZE-1))):
        N = [X[iy, ix-1], X[iy-1, ix-1], X[iy-1, ix], X[iy-1, ix+1], X[iy, ix+1]]

    # middle left boundary
    elif (((iy > 1) & (iy < GRID_SIZE-1)) & (ix == 1)):
        N = [X[iy-1, ix], X[iy-1, ix+1], X[iy, ix+1], X[iy+1, ix+1], X[iy+1, ix]]

    elif (((iy > 1) & (iy < GRID_SIZE-1)) & ((ix > 1) & (ix < GRID_SIZE-1))):
        N = [X[iy-1, ix], X[iy-1, ix+1], X[iy, ix+1], X[iy+1, ix+1], 
             X[iy+1, ix], X[iy+1, ix-1], X[iy, ix-1], X[iy-1, ix-1]]
        
    else: 
        raise Exception('We should never get here... Examine conditions above.')

    # return the number of currently burning nodes 
    return N.count(FIRE)



def iterate_fire_v3(X: np.array, phoschek_array: np.array, i: int):
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
        down_wind_spread_prob = EnvParams.down_wind_spread_prob
        
        # The boundary of the forest is always empty, so only consider cells
        # indexed from 1 to nx-2, 1 to ny-2

        X1 = np.zeros((ny, nx))
        nodes_searched = 0
        curr_burning_nodes = 0
        
        np.copyto(X1, X, where=X != EMPTY)

        # iterate over the currently burning nodes 
        for ix in range(1,nx):
            for iy in range(1,ny):
                if X[iy, ix] == TREE:
                    # get the number of burning neighbor nodes
                    b_i: int = get_num_burning_neighbors(iy=iy, ix=ix, X=X)

                    if b_i > 0: 
                        prob_fire: float = 1 - ((1 - ALPHA)**b_i)
                        if np.random.random() < prob_fire: 
                            X1[iy,ix] = FIRE
                
                if X[iy,ix] == FIRE:
                    curr_burning_nodes += 1
                    X1[iy,ix] = EMPTY

                # if X[iy,ix] == FIRE:
                #     curr_burning_nodes += 1
                #     X1[iy,ix] = EMPTY
                    
                #     # iterate over each of the 8 actions and their respective neighbor nodes
                #     # here we index by dy then dx since that's how Numpy indexing works 
                #     for action, (dy,dx) in action_to_direction.items():
                #         # print(f'action: {action}, direction: {direction_dict[wind]}, dx: {dx}, dy: {dy}')
                #         nodes_searched += 1
                #         # The diagonally-adjacent trees are further away, so
                #         # only catch fire with a reduced probability:
                #         if abs(dx) == abs(dy) and wind == 'none' and np.random.random() < 0.573:
                #             continue


                #         if wind != 'none' and np.random.random() < 0.03:
                #             continue
                        
                #         try: 
                #             if X[iy+dy,ix+dx] == TREE:
                #                 # in this case there is no wind so the fire spreads radially outwards
                #                 if wind == 'none': 
                #                     if np.random.random() < fire_spread_prob:
                #                         if ((phoschek_array[iy+dy,ix+dx] == 0) & (phoschek_array[iy+dy,ix] == 0) & (phoschek_array[iy,ix+dx] == 0)):
                #                             X1[iy+dy,ix+dx] = FIRE
                #                         # break
                #                 else:
                #                     # account for wind
                #                     if action==direction_dict[wind]: 
                #                         if ((phoschek_array[iy+dy,ix+dx] == 0) & (phoschek_array[iy+dy,ix] == 0) & (phoschek_array[iy,ix+dx] == 0)):
                #                             # this prevents straight right corners from occuring in the fire frontier
                #                             # TODO fix fire from going out 9% of the time 
                #                             if np.random.random() < down_wind_spread_prob:
                #                                 X1[iy+dy,ix+dx] = FIRE
                                    
                #                     # add additional condition to slow fire spread based on probability of spreading
                #                     else:
                #                         if np.random.random() < up_wind_spread_prob:
                #                             if ((phoschek_array[iy+dy,ix+dx] == 0) & (phoschek_array[iy+dy,ix] == 0) & (phoschek_array[iy,ix+dx] == 0)):
                #                                 X1[iy+dy,ix+dx] = FIRE
                #         except: 
                #             # here the fire reaches the boarder of the map so we can't index its neighbors! 
                #             # we add this try - except block to prevent index out of bounds errors
                #             pass
                            
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
    X[:ny, :nx] = np.random.randint(0, 2, size=(ny, nx))
    X[:ny, :nx] = np.random.random(size=(ny, nx)) < FOREST_FRACTION

    if IGNITION_POINTS == 1:
        # we only have one ignition point at the center of the environment
        ignition_index = int(GRID_SIZE/2)
        X[ignition_index, ignition_index] = FIRE
        # X[ignition_index+1, ignition_index] = FIRE
        # X[ignition_index, ignition_index+1] = FIRE
        # X[ignition_index-1, ignition_index] = FIRE
        # X[ignition_index, ignition_index-1] = FIRE
    else:
        # randomly select some ignition points
        point_list = [int(GRID_SIZE - GRID_SIZE/5), int(GRID_SIZE/2), int(GRID_SIZE/3), int(GRID_SIZE - GRID_SIZE/7), int(GRID_SIZE/9), int(GRID_SIZE/4)]
        for i in range(IGNITION_POINTS):
            X[np.random.choice(point_list), np.random.choice(point_list)] = FIRE

    return X


def main():

    tic = time.time()

    X = initialize_env()

    phoschek_array = np.zeros((EnvParams.grid_size, EnvParams.grid_size))

    frames = [X]

    # run the RL algorithm and get the frames of the environment state 
    done = False
    for i, step in enumerate(range(10_000)):
        X_dict = iterate_fire_v3(X=X, phoschek_array=phoschek_array, i=i)
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
    
    # plot or show the animation
    if AnimationParams.show_anim | AnimationParams.save_anim:
        plot_animation(frames=frames, repeat=AnimationParams.repeat, interval=AnimationParams.interval, 
                    save_anim=AnimationParams.save_anim, show_anim=AnimationParams.show_anim)
    

    # here we generate a mask showing the burned nodes for this run and return it
    # NOTE: we represent burned nodes using 1 here instead of the integer representing BURNED
    # so that we can generate multi-episode distributions of the burn area for each fire. 
    # make a copy of the final array 
    arr = X.copy()
    # multiply all non-zero elements by 2 
    arr = arr * 2
    # set all zero elements to 1
    arr[arr == 0] = 1
    # set all non-one elements to zero, leaving only the burned nodes as ones and all others as zeros
    arr[arr > 1] = 0
    
    return arr 


def get_burn_dist(save_fig: bool = False, show_fig: bool = True):
    """
    Function to generate the burn distribution 
    """
    # Don't save animations on each run because we will be running many animations here 
    AnimationParams.save_anim = False
    episodes: int = 3

    # generate subplots 
    # fig, axn = plt.subplots(3, 1, sharex=True, sharey=True)

    # for i, ax in enumerate(axn.flat):
    #     EnvParams.up_wind_spread_prob = (i+1)*0.06

    #     fire_mask_list = []
    #     for i in range(episodes):
    #         # set the random seed to a different value on each iteration
    #         np.random.seed(i)
    #         fire_mask: np.array = main()
    #         fire_mask_list.append(fire_mask)
        
    #     arr_sum: np.array = sum(fire_mask_list)
        
    #     # plot the resulting fire distribution
    #     sns.heatmap(arr_sum, linewidth=0, cmap="flare_r", ax=ax)
        # ax1.set(title=f'Burn Distribution Over {episodes} Episodes, $\omega_w$: {EnvParams.down_wind_spread_prob}, $\gamma_w$: {EnvParams.up_wind_spread_prob}')
    

    # # --------------- Parameter Setting #1 ---------------
    # EnvParams.up_wind_spread_prob = 0.12

    fire_mask_list = []
    for i in range(episodes):
        # set the random seed to a different value on each iteration
        np.random.seed(i)
        fire_mask: np.array = main()
        fire_mask_list.append(fire_mask)
    
    arr_sum: np.array = sum(fire_mask_list)
    
    # plot the resulting fire distribution
    ax1 = sns.heatmap(arr_sum, linewidth=0, cmap="flare_r")
    ax1.set(title=f'Burn Distribution Over {episodes} Episodes, $\omega_w$: {EnvParams.down_wind_spread_prob}, $\gamma_w$: {EnvParams.up_wind_spread_prob}')
    
    # # --------------- Parameter Setting #2 ---------------
    # EnvParams.up_wind_spread_prob = 0.12

    # fire_mask_list = []
    # for i in range(episodes):
    #     # set the random seed to a different value on each iteration
    #     np.random.seed(i)
    #     fire_mask: np.array = main()
    #     fire_mask_list.append(fire_mask)
    
    # arr_sum: np.array = sum(fire_mask_list)
    
    # # plot the resulting fire distribution
    # ax2 = sns.heatmap(arr_sum, linewidth=0, cmap="flare_r")
    # ax2.set(title=f'Burn Distribution Over {episodes} Episodes, $\omega_w$: {EnvParams.down_wind_spread_prob}, $\gamma_w$: {EnvParams.up_wind_spread_prob}')
    
    # # --------------- Parameter Setting #3 ---------------
    # EnvParams.up_wind_spread_prob = 0.16

    # fire_mask_list = []
    # for i in range(episodes):
    #     # set the random seed to a different value on each iteration
    #     np.random.seed(i)
    #     fire_mask: np.array = main()
    #     fire_mask_list.append(fire_mask)
    
    # arr_sum: np.array = sum(fire_mask_list)
    
    # # plot the resulting fire distribution
    # ax3 = sns.heatmap(arr_sum, linewidth=0, cmap="flare_r")
    # ax3.set(title=f'Burn Distribution Over {episodes} Episodes, $\omega_w$: {EnvParams.down_wind_spread_prob}, $\gamma_w$: {EnvParams.up_wind_spread_prob}')
    


    # save the resulting figure to disk
    if save_fig: 
        fig = ax1.get_figure()
        fig.savefig(f"burn_distribution_{episodes}_episodes.svg", format="svg") 
    
    # show the figure 
    if show_fig: 
        plt.show()


if __name__ == "__main__":
    main()
    # get_burn_dist(save_fig=True, show_fig=True)
