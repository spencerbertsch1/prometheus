from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import colors
import time 

tic = time.time()
# set the random seed for predictable runs 
np.random.seed(0)

# Displacements from a cell to its eight nearest neighbours
neighbourhood = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE, PLANE = 0, 1, 2, 3
# Colours for visualization: brown for EMPTY, dark green for TREE and orange
# for FIRE. Note that for the colormap to work, this list and the bounds list
# must be one larger than the number of different values in the array.
colors_list = ['black', 'green', 'darkorange', 'grey']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)


def numpy_element_counter(arr: np.array) -> dict:
    """
    Utility function to return a dict containing the counts for elements in np array
    """
    unique, counts = np.unique(arr, return_counts=True)
    counts_dict: dict = dict(zip(unique, counts))
    return counts_dict


def iterate(X, nx, ny, fire_spread_prob: float):
    """Iterate the forest according to the forest-fire rules."""

    # The boundary of the forest is always empty, so only consider cells
    # indexed from 1 to nx-2, 1 to ny-2
    X1 = np.zeros((ny, nx))
    cnt = 0
    
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            if X[iy,ix] == TREE:
                X1[iy,ix] = TREE
                for dx,dy in neighbourhood:
                    cnt += 1
                    # The diagonally-adjacent trees are further away, so
                    # only catch fire with a reduced probability:
                    if abs(dx) == abs(dy) and np.random.random() < 0.573:
                        continue
                    if X[iy+dy,ix+dx] == FIRE:
                        # add additional condition to slow fire spread based on probability of spreading
                        if np.random.random() < fire_spread_prob:
                            X1[iy,ix] = FIRE
                            break
    print(f'Nodes Processed: {cnt}')
    return X1

# The initial fraction of the forest occupied by trees.
forest_fraction = 0.9

# define fire spread probability
fire_spread_prob = 0.65
fire_speed = 0.8

# Forest size (number of cells in x and y directions).
grid_size = 200
nx, ny = grid_size, grid_size

ignition_points = 2

# Initialize the forest grid.
X  = np.zeros((ny, nx))
X[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
X[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < forest_fraction

if ignition_points == 1:
    # we only have one ignition point at the center of the environment
    ignition_index = int(grid_size/2)
    X[ignition_index, ignition_index] = FIRE
elif ignition_points == 2:
    # randomly select some ignition points
    point_list = [int(grid_size - grid_size/5), int(grid_size/2), int(grid_size/3), int(grid_size - grid_size/7), int(grid_size/9), int(grid_size/4)]
    X[np.random.choice(point_list), np.random.choice(point_list)] = FIRE
    X[np.random.choice(point_list), np.random.choice(point_list)] = FIRE


frames = []

# run the RL algorithm and get the frames of the environment state 
done = False
for step in range(10_000):
    X = iterate(X, nx=nx, ny=ny, fire_spread_prob=fire_spread_prob)
    frames.append(X)

    # test to see if there are any fire nodes remaining in the environment 
    node_cnt_dict = numpy_element_counter(arr=X)
    dict_keys = list(node_cnt_dict.keys())
    if FIRE not in dict_keys: 
        done = True

    if done:
        break

# define helper function
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat: bool, interval: int, save_anim: bool, show_anim: bool):
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
        anim.save('test_anim.mp4', writer='ffmpeg', fps=fps)

    if show_anim: 
        plt.show()

    return anim

repeat = False  # repeat the animation
interval = 100  # millisecond interval for each frame in the animation
save_anim = True  # save the animation to disk 
show_anim = False

plot_animation(frames, repeat=repeat, interval=interval, save_anim=True, show_anim=show_anim)
toc = time.time()
print(toc-tic)


"""
TODOs

- Only iterate over currently burning nodes (FIRE nodes)
- Add wind 
- Add multiple ignition points 



"""