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

# Create a forest fire animation based on a simple cellular automaton model.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-forest-fire-model/
# Christian Hill, January 2016.
# Updated January 2020.

# Displacements from a cell to its eight nearest neighbours
neighbourhood = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
EMPTY, TREE, FIRE = 0, 1, 2
# Colours for visualization: brown for EMPTY, dark green for TREE and orange
# for FIRE. Note that for the colormap to work, this list and the bounds list
# must be one larger than the number of different values in the array.
colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)

def iterate(X, nx, ny, fire_spread_prob: float, fire_speed: float):
    """Iterate the forest according to the forest-fire rules."""

    # The boundary of the forest is always empty, so only consider cells
    # indexed from 1 to nx-2, 1 to ny-2
    X1 = np.zeros((ny, nx))
    cnt = 0
    # Idea: only iterate through the burning nodes
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            if X[iy,ix] == TREE:
                X1[iy,ix] = TREE
    
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):  
            if X[iy,ix] == FIRE:
                X1[iy,ix] = FIRE
                for dx,dy in neighbourhood:
                    cnt += 1
                    # The diagonally-adjacent trees are further away, so
                    # only catch fire with a reduced probability:
                    if abs(dx) == abs(dy) and np.random.random() < 0.573:
                        continue
                    if X[iy+dy,ix+dx] == TREE:
                        # add additional condition to slow fire spread based on probability of spreading
                        if np.random.random() < fire_spread_prob:
                            X1[iy+dy,ix+dx] = FIRE
                            # break
    print(cnt)
    return X1

def main():

    # The initial fraction of the forest occupied by trees.
    # forest_fraction = 0.2
    forest_fraction = 0.9

    # define fire spread probability
    fire_spread_prob = 0.6
    fire_speed = 0.8

    # Forest size (number of cells in x and y directions).
    grid_size = 100
    nx, ny = grid_size, grid_size
    # Initialize the forest grid.
    X  = np.zeros((ny, nx))
    X[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
    X[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < forest_fraction

    # set initial fire ignition location
    start_index = int(grid_size/2)
    X[start_index, start_index] = FIRE

    fig = plt.figure(figsize=(25/3, 6.25))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(X, cmap=cmap, norm=norm)#, interpolation='nearest')

    # The animation function: called to produce a frame for each generation.
    def animate(i):
        im.set_data(animate.X)
        animate.X = iterate(animate.X, nx=nx, ny=ny, fire_spread_prob=fire_spread_prob, fire_speed=fire_speed)
    # Bind our grid to the identifier X in the animate function's namespace.
    animate.X = X

    # Interval between frames (ms).
    interval = 10
    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=200)
    plt.show()


if __name__ == "__main__":
    main()


"""
TODOs 

1. Improve efficiency by iterating through only the burning nodes. 

"""