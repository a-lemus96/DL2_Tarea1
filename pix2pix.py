import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import datetime
from IPython import display


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Definition of global variables
INPUT_DIM = (256, 256, 3)
OUTPUT_CHANNELS = INPUT_DIM[-1]
BATCH_SIZE = 10
R_LOSS_FACTOR = 10000
EPOCHS = 100
INITIAL_EPOCH = 0

X_LEFT_FOLDER = "/home/est_posgrado_alejandro.lemus/DL2/Tarea1/training_data/left_X"
X_RIGHT_FOLDER = "/home/est_posgrado_alejandro.lemus/DL2/Tarea1/training_data/right_X"
Y_TARGET_FOLDER = "/home/est_posgrado_alejandro.lemus/DL2/Tarea1/training_data/target_Y"

# Create lists of file directories
left_x_files = glob(os.path.join(X_LEFT_FOLDER, '*.png')).sort()
right_x_files = glob(os.path.join(X_RIGHT_FOLDER, '*.png')).sort()
target_y_files = glob(os.path.join(Y_TARGET_FOLDER, '*.png')).sort()

for lx, rx, y in zip(left_x_files[:5], right_x_files[:5], target_y_files[:5]):
    print(f"{lx}\n{rx}\n{y}\n\n")