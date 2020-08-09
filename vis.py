from model import *
from utils import *
import torch
import matplotlib.pyplot as plt
import gif
from matplotlib.animation import FuncAnimation
import numpy as np

"""

USING MATPLOTLIBT TO VISUZLIZE WHAT'S HAPPENING INSIZE OUR MODEL WHILE TRAINING?


"""

x,y = [],[]
@gif.frame
def plot(i):
    plt.plot(torch.arange(i),
    torch.arange(i)*np.exp(i))
frames = [plot(i) for i in range(100)]
gif.save(frames, 'torch.gif', duration=10)
