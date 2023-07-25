import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import time
from random import choices
import seaborn as sns

def plot_np_array():
    """
    https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    """
    np.random.seed(101)
    zvals = np.random.rand(100, 100) * 5

    # make a color map of fixed colors
    # https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'orange'])
    bounds=[0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    img = plt.imshow(zvals, interpolation='nearest', origin='lower',
                        cmap=cmap, norm=norm)

    # make a color bar
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])

    plt.show()




def timer():
    A = [0, 1, 2, 3, 4, 5, 6, 7]

    tic = time.time()
    l = choices(A, k=1_000_000)
    print(f'Time taken for method 1: {time.time() - tic}')

    tic = time.time()
    l = [random.choice(A) for x in range(1_000_000)]
    print(f'Time taken for method 2: {time.time() - tic}')


def get_centroid():
    arr = np.zeros([10, 10])
    for i in range(10):
        x = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        y = np.random.choice([3, 4, 5, 6, 7])
        arr[x, y] = 1
    ax = sns.heatmap(arr, linewidth=0.5)
    plt.show()

    count = (arr == 1).sum()
    y_center, x_center = np.argwhere(arr==1).sum(0)/count
    print(x_center, y_center)

if __name__ == "__main__":
    get_centroid()


