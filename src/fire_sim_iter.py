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
from matplotlib.animation import FuncAnimation 
from matplotlib import animation
   
# # initializing a figure in 
# # which the graph will be plotted
# fig = plt.figure() 
   
# # marking the x-axis and y-axis
# axis = plt.axes(xlim =(0, 4), 
#                 ylim =(-2, 2)) 
  
# # initializing a line variable
# line, = axis.plot([], [], lw = 3) 
   
# # data which the line will 
# # contain (x, y)
# def init(): 
#     line.set_data([], [])
#     return line,
   
# def animate(i):
#     x = np.linspace(0, 4, 1000)
   
#     # plots a sine graph
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
      
#     return line,
   
# anim = FuncAnimation(fig, animate, init_func = init,
#                      frames = 200, interval = 20)
  
# plt.show()



# anim.save('continuousSineWave.mp4', 
#           writer = 'ffmpeg', fps = 30)

fig = plt.figure() 
   
# marking the x-axis and y-axis
axis = plt.axes(xlim =(0, 4), 
                ylim =(-2, 2)) 

frames = []
line, = axis.plot([], [], lw = 3) 

done = False
for step in range(200):
    rand_arr = np.random.rand(3, 3)
    frames.append(rand_arr)
    if step > 20: 
        done = True

    if done:
        break

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.show()
    # plt.close()

    return anim

plot_animation(frames)