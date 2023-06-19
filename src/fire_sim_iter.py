"""
--------------------------------------
| Wildfire Containment Research      |
| Dartmouth, 2023                    |
| Spencer Bertsch                    |
--------------------------------------

This script was adapted from: 
https://scipython.com/blog/the-forest-fire-model/

https://thecleverprogrammer.com/2020/07/26/openai-gym-in-machine-learning/

"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
   
frames = []

# run the RL algorithm and get the frames of the environment state 
done = False
for step in range(200):
    rand_arr = np.random.rand(3, 3)
    frames.append(rand_arr)
    if step > 20: 
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
show_anim = True
plot_animation(frames, repeat=repeat, interval=interval, save_anim=True, show_anim=show_anim)