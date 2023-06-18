"""
--------------------------------------
| Wildfire Containment Research      |
| Dartmouth, 2023                    |
| Spencer Bertsch                    |
--------------------------------------

This script was adapted from: 
https://scipython.com/blog/the-forest-fire-model/

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors

# set the random seed for predictable runs 
np.random.seed(0)

# Create a forest fire animation based on a simple cellular automaton model.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-forest-fire-model/
# Christian Hill, January 2016.
# Updated January 2020.

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
    print(cnt)
    return X1

def random_action(old_agent_location):
    move_index = np.random.choice(range(len(neighbourhood)))
    move = neighbourhood[move_index]
    new_agent_location = [old_agent_location[0] + move[0], old_agent_location[1] + move[1]]
    return new_agent_location

def main():

    # The initial fraction of the forest occupied by trees.
    forest_fraction = 0.9

    # define fire spread probability
    fire_spread_prob = 0.65
    fire_speed = 0.8

    # Forest size (number of cells in x and y directions).
    grid_size = 50
    nx, ny = grid_size, grid_size
    # Initialize the forest grid.
    X  = np.zeros((ny, nx))
    X[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
    X[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < forest_fraction

    # set initial fire ignition location
    ignition_index = int(grid_size/2)
    X[ignition_index, ignition_index] = FIRE

    # set initial aircraft location
    plane_start_index = grid_size-10
    agent_location = [plane_start_index, plane_start_index]
    X[plane_start_index, plane_start_index] = PLANE

    fig = plt.figure(figsize=(25/3, 6.25))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    # im = ax.imshow(X, cmap=cmap, norm=norm)#, interpolation='nearest')
    im = ax.imshow(X, cmap=cmap, interpolation='nearest')

    # The animation function: called to produce a frame for each generation.
    def animate(i):
        im.set_data(animate.X)
        animate.X = iterate(animate.X, nx=nx, ny=ny, fire_spread_prob=fire_spread_prob)
    
    # Bind our grid to the identifier X in the animate function's namespace.
    # move the agent
    agent_location = random_action(old_agent_location=agent_location)
    X[agent_location[0], agent_location[1]] = PLANE
    animate.X = X

    # Interval between frames (ms).
    interval = 100
    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=200)
    plt.show()


if __name__ == "__main__":
    main()


"""
TODOs 

1. Improve efficiency by iterating through only the burning nodes. 

"""