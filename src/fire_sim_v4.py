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
from scipy.signal import convolve2d

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
ALPHA = EnvParams.alpha
GRID_SIZE = EnvParams.grid_size
IGNITION_POINTS = EnvParams.ignition_points
# define the three parameters that control fire dynamics
W = EnvParams.w
DELTA = EnvParams.delt
EPSILON = EnvParams.epsilon

# set the random seed for predictable runs 
np.random.seed(1)

# Colours for visualization: brown for EMPTY, dark green for TREE and orange
# for FIRE. Note that for the colormap to work, this list and the bounds list
# must be one larger than the number of different values in the array.
colors_list = ['black', 'green', 'darkorange', 'grey', 1]
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)

def get_num_burning_neighbors(iy: int, ix: int, X: np.array, delta: float):
    """
    OLD CODE
    ------------------------------------------------------------------------
    THIS IS AN OLD FUNCTION THAT HAS BEEN REPLACED BY THE FOLLOWING FUNCTION: 
    get_b_wind()

    THIS FUNCTION (get_num_burning_neighbors) WILL BE REMOVED AFTER ALL FIRE 
    SPREAD FUNCTIONS HAVE BEEN FINALIZED. 
    """

    # upper left corner 
    if ((iy==0) & (ix == 0)):
        N = [X[iy+1, ix], X[iy, ix+1]]
        N_diag = [X[iy+1, ix+1]]

    # upper right corner 
    elif ((iy==0) & (ix == GRID_SIZE-1)):
        N = [X[iy+1, ix], X[iy, ix-1]]
        N_diag = [X[iy+1, ix-1]]

    # lower left corner 
    elif ((iy==GRID_SIZE-1) & (ix == 0)):
        N = [X[iy-1, ix], X[iy, ix+1]]
        N_diag = [X[iy-1, ix+1]]
    
    # lower right corner 
    elif ((iy==GRID_SIZE-1) & (ix == GRID_SIZE-1)):
        N = [X[iy-1, ix], X[iy, ix-1]]
        N_diag = [X[iy-1, ix-1]]

    # middle upper boundary
    elif ((iy == 0) & ((ix > 0) & (ix < GRID_SIZE-1))):
        N = [X[iy, ix-1], X[iy+1, ix], X[iy, ix+1]]
        N_diag = [X[iy+1, ix-1], X[iy+1, ix+1]]

    # middle right boundary
    elif (((iy > 0) & (iy < GRID_SIZE-1)) & (ix == GRID_SIZE-1)):
        N = [X[iy-1, ix],  X[iy, ix-1], X[iy+1, ix]]
        N_diag = [X[iy-1, ix-1], X[iy+1, ix-1]]

    # middle bottom boundary
    elif ((iy == GRID_SIZE-1) & ((ix > 0) & (ix < GRID_SIZE-1))):
        N = [X[iy, ix-1],  X[iy-1, ix],  X[iy, ix+1]]
        N_diag = [X[iy-1, ix-1], X[iy-1, ix+1]]

    # middle left boundary
    elif (((iy > 0) & (iy < GRID_SIZE-1)) & (ix == 0)):
        N = [X[iy-1, ix],  X[iy, ix+1],  X[iy+1, ix]]
        N_diag = [X[iy-1, ix+1], X[iy+1, ix+1]]

    # non-boundary node 
    elif (((iy > 0) & (iy < GRID_SIZE-1)) & ((ix > 0) & (ix < GRID_SIZE-1))):
        N = [X[iy-1, ix], X[iy, ix+1], X[iy+1, ix], X[iy, ix-1]]
        N_diag = [X[iy-1, ix+1], X[iy+1, ix+1], X[iy+1, ix-1], X[iy-1, ix-1]]
        
    else: 
        raise Exception('We should never get here... Examine conditions above.')

    # return the number of currently burning nodes 
    return N.count(FIRE) + (N_diag.count(FIRE) * delta)


def get_b_wind_eight_neighbor(iy: int, ix: int, X: np.array, delta: float):

    # ensure the wind is one of the expected values 
    assert WIND in ['none', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], \
        f'Wind should be \'none\', \'N\', ..., \'NW\', not {WIND}'

    if WIND == "none":

        # upper left corner 
        if ((iy==0) & (ix == 0)):
            N = [X[iy+1, ix], X[iy, ix+1]]
            N_diag = [X[iy+1, ix+1]]

        # upper right corner 
        elif ((iy==0) & (ix == GRID_SIZE-1)):
            N = [X[iy+1, ix], X[iy, ix-1]]
            N_diag = [X[iy+1, ix-1]]

        # lower left corner 
        elif ((iy==GRID_SIZE-1) & (ix == 0)):
            N = [X[iy-1, ix], X[iy, ix+1]]
            N_diag = [X[iy-1, ix+1]]
        
        # lower right corner 
        elif ((iy==GRID_SIZE-1) & (ix == GRID_SIZE-1)):
            N = [X[iy-1, ix], X[iy, ix-1]]
            N_diag = [X[iy-1, ix-1]]

        # middle upper boundary
        elif ((iy == 0) & ((ix > 0) & (ix < GRID_SIZE-1))):
            N = [X[iy, ix-1], X[iy+1, ix], X[iy, ix+1]]
            N_diag = [X[iy+1, ix-1], X[iy+1, ix+1]]

        # middle right boundary
        elif (((iy > 0) & (iy < GRID_SIZE-1)) & (ix == GRID_SIZE-1)):
            N = [X[iy-1, ix],  X[iy, ix-1], X[iy+1, ix]]
            N_diag = [X[iy-1, ix-1], X[iy+1, ix-1]]

        # middle bottom boundary
        elif ((iy == GRID_SIZE-1) & ((ix > 0) & (ix < GRID_SIZE-1))):
            N = [X[iy, ix-1],  X[iy-1, ix],  X[iy, ix+1]]
            N_diag = [X[iy-1, ix-1], X[iy-1, ix+1]]

        # middle left boundary
        elif (((iy > 0) & (iy < GRID_SIZE-1)) & (ix == 0)):
            N = [X[iy-1, ix],  X[iy, ix+1],  X[iy+1, ix]]
            N_diag = [X[iy-1, ix+1], X[iy+1, ix+1]]

        # non-boundary node 
        elif (((iy > 0) & (iy < GRID_SIZE-1)) & ((ix > 0) & (ix < GRID_SIZE-1))):
            N = [X[iy-1, ix], X[iy, ix+1], X[iy+1, ix], X[iy, ix-1]]
            N_diag = [X[iy-1, ix+1], X[iy+1, ix+1], X[iy+1, ix-1], X[iy-1, ix-1]]
            
        else: 
            raise Exception('We should never get here... Examine conditions above.')
        
    elif WIND == "N":
        N = [X[iy+1, ix]]
        N_diag = [X[iy+1, ix+1], X[iy+1, ix-1]]

    elif WIND == "NE":
        N = [X[iy+1, ix], X[iy, ix-1]]
        N_diag = [X[iy+1, ix-1]]

    elif WIND == "E":
        N = [X[iy, ix-1]]
        N_diag = [X[iy+1, ix-1], X[iy-1, ix-1]]

    elif WIND == "SE":
        N = [X[iy, ix-1], X[iy-1, ix]]
        N_diag = [X[iy-1, ix-1]]

    elif WIND == "S":
        N = [X[iy-1, ix]]
        N_diag = [X[iy-1, ix-1], X[iy-1, ix+1]]

    elif WIND == "SW":
        N = [X[iy-1, ix], X[iy, ix+1]]
        N_diag = [X[iy-1, ix+1]]

    elif WIND == "W":
        N = [X[iy, ix+1]]
        N_diag = [X[iy-1, ix+1], X[iy+1, ix+1]]

    elif WIND == "NW":
        N = [X[iy, ix+1], X[iy+1, ix]]
        N_diag = [X[iy+1, ix+1]]

    else: 
        raise Exception('We should never get here... Examine conditions above.')
    
            # define the wind coefficient based on the cosine of the wind direction 
    lamda = math.cos(math.radians(45)) * W

    # write the logic in section 2.2.1 in the wildfires paper (wind fire transition probability)
    if WIND != 'none':
        if direction_dict[WIND] % 2 == 0:
            # the direction is even, so the wind is N, E, S, or W
            b_ij: float = N.count(FIRE) + (N_diag.count(FIRE) * (DELTA * lamda)) + EPSILON
        else:
            # the direction is odd, so the wind is NE, SE, SW, or NW
            b_ij: float = (N.count(FIRE) * lamda) + ((N_diag.count(FIRE) * DELTA)) + EPSILON
    else:
        # there is no wind here, so we dont need to worry about w (wind speed), lambda (wind coefficient) or 
        # espilon (small constant)
        b_ij: float = N.count(FIRE) + (N_diag.count(FIRE) * DELTA)

    # return the number of currently burning nodes 
    return b_ij
    # return N.count(FIRE) + (N_diag.count(FIRE) * (delta * lamda)) + epsilon


def get_b_wind_four_neighbor(iy: int, ix: int, X: np.array, delta: float):

    # ensure the wind is one of the expected values 
    assert WIND in ['none', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], \
        f'Wind should be \'none\', \'N\', ..., \'NW\', not {WIND}'

    if WIND == "none":

        # upper left corner 
        if ((iy==0) & (ix == 0)):
            N = [X[iy+1, ix], X[iy, ix+1]]

        # upper right corner 
        elif ((iy==0) & (ix == GRID_SIZE-1)):
            N = [X[iy+1, ix], X[iy, ix-1]]

        # lower left corner 
        elif ((iy==GRID_SIZE-1) & (ix == 0)):
            N = [X[iy-1, ix], X[iy, ix+1]]
        
        # lower right corner 
        elif ((iy==GRID_SIZE-1) & (ix == GRID_SIZE-1)):
            N = [X[iy-1, ix], X[iy, ix-1]]

        # middle upper boundary
        elif ((iy == 0) & ((ix > 0) & (ix < GRID_SIZE-1))):
            N = [X[iy, ix-1], X[iy+1, ix], X[iy, ix+1]]

        # middle right boundary
        elif (((iy > 0) & (iy < GRID_SIZE-1)) & (ix == GRID_SIZE-1)):
            N = [X[iy-1, ix],  X[iy, ix-1], X[iy+1, ix]]

        # middle bottom boundary
        elif ((iy == GRID_SIZE-1) & ((ix > 0) & (ix < GRID_SIZE-1))):
            N = [X[iy, ix-1],  X[iy-1, ix],  X[iy, ix+1]]

        # middle left boundary
        elif (((iy > 0) & (iy < GRID_SIZE-1)) & (ix == 0)):
            N = [X[iy-1, ix],  X[iy, ix+1],  X[iy+1, ix]]

        # non-boundary node 
        elif (((iy > 0) & (iy < GRID_SIZE-1)) & ((ix > 0) & (ix < GRID_SIZE-1))):
            N = [X[iy-1, ix], X[iy, ix+1], X[iy+1, ix], X[iy, ix-1]]
            
        else: 
            raise Exception('We should never get here... Examine conditions above.')
        
    elif WIND == "N":
        N = [X[iy+1, ix]]

    elif WIND == "NE":
        N = [X[iy+1, ix], X[iy, ix-1]]

    elif WIND == "E":
        N = [X[iy, ix-1]]

    elif WIND == "SE":
        N = [X[iy, ix-1], X[iy-1, ix]]

    elif WIND == "S":
        N = [X[iy-1, ix]]

    elif WIND == "SW":
        N = [X[iy-1, ix], X[iy, ix+1]]

    elif WIND == "W":
        N = [X[iy, ix+1]]

    elif WIND == "NW":
        N = [X[iy, ix+1], X[iy+1, ix]]

    else: 
        raise Exception('We should never get here... Examine conditions above.')
    
            # define the wind coefficient based on the cosine of the wind direction 
    lamda = math.cos(math.radians(45)) * W

    # write the logic in section 2.2.1 in the wildfires paper (wind fire transition probability)
    if WIND != 'none':
        if direction_dict[WIND] % 2 ==0:
            # the direction is even, so the wind is N, E, S, or W
            b_ij: float = (N.count(FIRE) * lamda) + EPSILON
        else:
            # the direction is odd, so the wind is NE, SE, SW, or NW
            b_ij: float = (N.count(FIRE) * lamda) + EPSILON
    else:
        # there is no wind here, so we dont need to worry about w (wind speed), lambda (wind coefficient) or 
        # espilon (small constant)
        b_ij: float = N.count(FIRE)

    # return the number of currently burning nodes 
    return b_ij
    # return N.count(FIRE) + (N_diag.count(FIRE) * (delta * lamda)) + epsilon

def test_get_fire_adjacent_nodes():
    """
    Tester for get_fire_adjacent_nodes function
    """
    X = np.array([[3, 1, 1, 3, 1],
                  [1, 2, 2, 2, 2],
                  [1, 2, 1, 1, 1],
                  [1, 2, 1, 0, 1],
                  [1, 2, 0, 3, 1]])
    
    adjacent_burn_indices = get_fire_adjacent_nodes(X=X)

    B = np.zeros([5, 5])
    for y, x in zip(adjacent_burn_indices[0], adjacent_burn_indices[1]):
        # print(f'Burn adjacent node: y:{y}, x:{x}')
        B[y, x] = 1

    solution = np.array([[1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0],
                         [1, 0, 1, 1, 1],
                         [1, 0, 1, 0, 0],
                         [1, 0, 1, 0, 0]])
    
    # test to see if the solutions match 
    assert np.array_equal(B, solution), f"TEST FAILED: \n {B} \n \n IS NOT EQUAL TO \n \n {solution}. \
                              Check get_fire_adjacent_nodes function. "
    print('TEST PASSED.')


def get_fire_adjacent_nodes(X: np.array):
    """
    Helper function that returns the indices of the nodes adjacent to fire nodes

    Based on: 
    https://stackoverflow.com/questions/65297426/get-indices-for-all-elements-adjacent-to-zeros-in-a-numpy-2d-array
    """
    # convert all non-fire nodes into tree nodes
    X[X != FIRE] = TREE
    
    kernel = np.full((3,3), 1)

    # remove center of kernel -- not count 1 at the center of the square
    # we may not need to remove center
    # in which case change the mask for counts
    kernel[1,1]=2

    # counts 1 among the neighborhoods
    counts = convolve2d(X, kernel, mode='same', 
                        boundary='fill', fillvalue=1)

    # counts==8 meaning surrounding 8 neighborhoods are all 1
    # change to 9 if we leave kernel[1,1] == 1
    # and we want ocean, i.e. a==1
    adjacent_burn_indices: tuple = np.where((counts != 10) & (X==1))

    if EnvParams.four_neighbor_burn: 
        """
        Here we need to remove the healthy nodes that only have diagonally adjacent fire nodes
        TODO - vectorize this section of the function ! 
        """
        four_neighbor_adjacent_burn_indices = []
        iy_index_list = []
        ix_index_list = []
        for iy, ix in zip(adjacent_burn_indices[0], adjacent_burn_indices[1]):
            # only keep the healthy nodes that only have vertically or horizontally adjacent fire nodes
            # FIXME remove the try except! This section is throwing an out-of-bounds error when the fire
            # reaches the perimiter of the env. Fix this later (try-except-pass works for now, although 
            # is not best practice programatically)
            try: 
                if ((X[iy+1, ix] == FIRE) | (X[iy-1, ix] == FIRE) | (X[iy, ix+1] == FIRE) | (X[iy, ix-1] == FIRE)):
                    iy_index_list.append(iy)
                    ix_index_list.append(ix)
            except:
                pass

        four_neighbor_adjacent_burn_indices = [tuple(iy_index_list), tuple(ix_index_list)]
        adjacent_burn_indices = tuple(four_neighbor_adjacent_burn_indices)

    return adjacent_burn_indices
    

def iterate_fire_v4(X: np.array, phoschek_array: np.array, i: int):
    """
    Iterate the forest according to the forest-fire rules.
    https://scipython.com/blog/the-forest-fire-model/
    
    https://stackoverflow.com/questions/63661231/replacing-numpy-array-elements-that-are-non-zero
    ^ This greatly accelerated the run time of this function
    """
   
    if i % EnvParams.fire_speed == 0:

        # define environment parameters based on the current configuration
        ny, nx = EnvParams.grid_size, EnvParams.grid_size
        
        # The boundary of the forest is always empty, so only consider cells
        # indexed from 1 to nx-2, 1 to ny-2

        X1 = np.zeros((ny, nx))
        nodes_searched = 0
        curr_burning_nodes = 0
        delta: float = 0.525
        
        np.copyto(X1, X, where=X != EMPTY)

        # get two tuples each containing the indices of all fire-adjacent nodes
        adjacent_burn_indices: tuple = get_fire_adjacent_nodes(X=X)

        for iy, ix in zip(adjacent_burn_indices[0], adjacent_burn_indices[1]):
            # print(f'Burn adjacent node: y:{iy}, x:{ix}')
            nodes_searched += 1

            if X1[iy, ix] == TREE:
                # get the number of burning neighbor nodes
                b_i: int = get_b_wind_eight_neighbor(iy=iy, ix=ix, X=X, delta=delta)

                if b_i > 0: 
                    prob_fire: float = 1 - ((1 - ALPHA)**b_i)
                    if np.random.random() < prob_fire:                       
                        """
                        Here we add logic to prevent fire from burning through diagonally placed fire retardant:
                        """
                        # Vertical or Horizontal burning neighbor case: 
                        # if any of the vertical or horizontal nodes are Fire nodes then we continue
                        # FIXME - remove this try-except-pass (replace with index checker) !
                        try: 
                            if ((X1[iy-1,ix] == FIRE) | (X1[iy+1,ix] == FIRE) | (X1[iy,ix-1] == FIRE) | (X1[iy,ix+1] == FIRE)):
                                X1[iy,ix] = FIRE
                            # Diagonal burning neighbor case (upper left): 
                            if X1[iy-1,ix-1] == FIRE:
                                if not ((phoschek_array[iy-1,ix] == PHOSCHEK) & (phoschek_array[iy,ix-1] == PHOSCHEK)):
                                    X1[iy,ix] = FIRE
                            # Diagonal burning neighbor case (upper right): 
                            if X1[iy-1,ix+1] == FIRE:
                                if not ((phoschek_array[iy-1,ix] == PHOSCHEK) & (phoschek_array[iy,ix+1] == PHOSCHEK)):
                                    X1[iy,ix] = FIRE
                            # Diagonal burning neighbor case (lower right): 
                            if X1[iy+1,ix+1] == FIRE:
                                if not ((phoschek_array[iy+1,ix] == PHOSCHEK) & (phoschek_array[iy,ix+1] == PHOSCHEK)):
                                    X1[iy,ix] = FIRE
                            # Diagonal burning neighbor case (lower left): 
                            if X1[iy+1,ix-1] == FIRE:
                                if not ((phoschek_array[iy+1,ix] == PHOSCHEK) & (phoschek_array[iy,ix-1] == PHOSCHEK)):
                                    X1[iy,ix] = FIRE
                        except: 
                            pass
                        
        # replace currently burning nodes with empty nodes 
        fire_indices: tuple = np.where(X == FIRE)
        curr_burning_nodes = len(fire_indices[0])
        X1[fire_indices] = EMPTY

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
        # use addiitonal starting burn nodes if wind is present so the fire doesnt burn out
        if EnvParams.wind != 'none':
            X[ignition_index+1, ignition_index] = FIRE
            X[ignition_index, ignition_index+1] = FIRE
            X[ignition_index-1, ignition_index] = FIRE
            X[ignition_index, ignition_index-1] = FIRE
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
        X_dict = iterate_fire_v4(X=X, phoschek_array=phoschek_array, i=i)
        X = X_dict['X']
        frames.append(X.copy())

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
    ax1.set(title=fr"""Burn Distribution Over {episodes} Episodes 
            Wind Direction: {EnvParams.wind}, $\delta$: {EnvParams.delt}, $\alpha$: {EnvParams.alpha}, $\epsilon$: {EnvParams.epsilon}, $w$: {EnvParams.w}""")
    
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
    # test_get_fire_adjacent_nodes()

