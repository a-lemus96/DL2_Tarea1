import os
from glob import glob
from random import shuffle
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
OUTPUT_CHANNELS = 1 # Grayscale image representing disparity
BATCH_SIZE = 1
R_LOSS_FACTOR = 10000
EPOCHS = 100
INITIAL_EPOCH = 0

X_LEFT_FOLDER = "training_data/left_X/"
X_RIGHT_FOLDER = "training_data/right_X/"
Y_TARGET_FOLDER = "training_data/target_Y/"

# Create lists of file directories
left_x_files = glob(os.path.join(X_LEFT_FOLDER, '*.png'))
right_x_files = glob(os.path.join(X_RIGHT_FOLDER, '*.png'))
target_y_files = glob(os.path.join(Y_TARGET_FOLDER, '*.png'))

# Sort elements
left_x_files.sort()
right_x_files.sort()
target_y_files.sort()

# Validation of lists of directories
print("First 5 triplets of files:\n")
for lx, rx, y in zip(left_x_files[:5], right_x_files[:5], target_y_files[:5]):
    print(f"{lx}\n{rx}\n{y}\n\n")

# Global variables for TF data loader
BUFFER_SIZE = len(left_x_files)
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
print(f"Number of image triplets: {BUFFER_SIZE}")
print(f"Steps per epoch: {steps_per_epoch}")


# Function definition for loading files and output tensors
def load_images(left_file, right_file, target_file):
    # Read and decode image files to uint8 tensors
    lx = tf.io.read_file(left_file)
    rx = tf.io.read_file(right_file)
    y = tf.io.read_file(target_file)

    lx = tf.io.decode_png(lx)
    rx = tf.io.decode_png(rx)
    y = tf.io.decode_png(y)

    # Convert the triplet of images to float32 tensors
    lx = tf.cast(lx, tf.float32)
    rx = tf.cast(rx, tf.float32)
    y = tf.cast(y, tf.float32)

    # Resize images
    lx = tf.image.resize(lx, [256, 256])
    rx = tf.image.resize(rx, [256, 256])
    y = tf.image.resize(y, [256, 256])

    # Normalize images
    lx = (lx / 127.5) - 1
    rx = (rx / 127.5) - 1
    y = (y / 127.5) - 1

    # Concatenate input images to form a HxWx6 tensor
    x = tf.concat([lx, rx], axis=-1)
    
    return x, y


# Function for saving images on a .png file for further consulting
def display_images(x_imgs=None, y_imgs=None, rows=4, cols=1, fname='output'):
    plt.figure(figsize=(cols*6, rows*2))
    for k in range(rows*cols):

        plt.subplot(rows, cols*3, 3*k + 1)
        plt.imshow(((x_imgs[k])[:, :, :3] + 1)/2) # Left-view
        plt.axis('off')

        plt.subplot(rows, cols*3, 3*k + 2)
        plt.imshow(((x_imgs[k])[:, :, 3:] + 1)/2) # Right-view
        plt.axis('off')

        plt.subplot(rows, cols*3, 3*k + 3)
        plt.imshow((y_imgs[k] + 1)/2, cmap='gray') # Target
        plt.axis('off')
    
    plt.savefig(f'{fname}.png')

x_imgs = []
y_imgs = []

# Sample 4 image triplets
for i in np.random.randint(0, len(left_x_files), 4):
    xim, yim = load_images(left_x_files[i], right_x_files[i], target_y_files[i])
    x_imgs.append(xim)
    y_imgs.append(yim)

# Save the image triplets within a .png file
display_images(x_imgs, y_imgs, fname='display_test')


# Creation of atomic training datasets
idx = int(BUFFER_SIZE*.8)

train_lx = tf.data.Dataset.list_files(left_x_files[:idx], shuffle=False)
train_rx = tf.data.Dataset.list_files(right_x_files[:idx], shuffle=False)
train_y = tf.data.Dataset.list_files(target_y_files[:idx], shuffle=False)

# Pairing previous datasets on a higher-level training dataset
train_lx_rx_y = tf.data.Dataset.zip((train_lx, train_rx, train_y))
train_lx_rx_y = train_lx_rx_y.shuffle(buffer_size=idx, reshuffle_each_iteration=True)
train_lx_rx_y = train_lx_rx_y.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_lx_rx_y = train_lx_rx_y.batch(BATCH_SIZE)

# Creation of atomic test datasets
test_lx = tf.data.Dataset.list_files(left_x_files[idx:], shuffle=False)
test_rx = tf.data.Dataset.list_files(right_x_files[idx:], shuffle=False)
test_y = tf.data.Dataset.list_files(target_y_files[idx:], shuffle=False)

# Pairing previous datasets on a higher-level test dataset
test_lx_rx_y = tf.data.Dataset.zip((test_lx, test_rx, test_y))
test_lx_rx_y = test_lx_rx_y.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
test_lx_rx_y = test_lx_rx_y.batch(BATCH_SIZE)

# Checking that Dataset object reads synchronized input/output pairs 
for x, y in train_lx_rx_y.take(1):
    display_images(x, y,  fname='dataset_test', rows=min(4, BATCH_SIZE))
    break

