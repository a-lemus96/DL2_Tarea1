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
INPUT_DIM = (256, 256, 6)
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


# Down-sampling block
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# Downsampling code verification
down_model = downsample(3,4)
down_result = down_model(tf.expand_dims(x_imgs[0], 0))

# Up-sampling block
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters=filters,
                                               kernel_size=size,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

# Upsampling verification
up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

# UNet Generator
def Generator():
    # Layers that compose the net
    x_input = tf.keras.layers.Input(shape=INPUT_DIM)
    down_stack = [
        downsample(64,  4, apply_batchnorm=False),# (batch_size, 128, 128, 64)
        downsample(128, 4),                       # (batch_size, 64,  64,  128)
        downsample(256, 4),                       # (batch_size, 32,  32,  256)
        downsample(512, 4),                       # (batch_size, 16,  16,  512)
    ]

    up_stack = [
        upsample(256, 4, apply_dropout=True),       # (batch_size, 32,   32, 512)
        upsample(128, 4, apply_dropout=True),       # (batch_size, 64,   64, 256)
        upsample(64,  4),                           # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                           4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (batch_size, 256, 256, 1)

    # Processing pipeline
    x = x_input
    
    # Encoder
    skips=[]
    for down in down_stack:
        x = down(x)
        skips.append(x) # Output for each downsampling is added to a list

    skips = reversed(skips[:-1])
    
    #Decoder
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=x_input, outputs=x)

# Test generator 
generator = Generator()
gen_output = generator(x_imgs[0][tf.newaxis, ...], training=False)
print(gen_output.shape)
plt.figure(figsize=(6.5,6.5))
plt.imshow(gen_output[0, ...]*50, cmap='gray')
plt.axis('off')
plt.savefig("test_generator.png")

# Discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 6], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar],  axis=-1)

    down1 = downsample(64, 4, False)(x) # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)    # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)    # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(filters=1, 
                                  kernel_size=4, 
                                  strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
disc_out = discriminator([x_imgs[0][tf.newaxis, ...], gen_output], training=False)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(gen_output[0, ...]*50, cmap='gray')
plt.subplot(122)
plt.imshow(disc_out[0, ..., -1]*200, vmin=-20, vmax=20, cmap='RdBu_r')  #*100
plt.colorbar()
plt.savefig('test_discriminator.png')


# Generator loss
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

# Discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Training

log_dir="logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Train step function definition
@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output      = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients     = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,     generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        with summary_writer.as_default():
            ss = step//1000
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=ss)
            tf.summary.scalar('gen_gan_loss',   gen_gan_loss,   step=ss)
            tf.summary.scalar('gen_l1_loss',    gen_l1_loss,    step=ss)
            tf.summary.scalar('disc_loss',      disc_loss,      step=ss)


def fit(train_ds, test_ds, steps):
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      #generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)


fit(train_lx_rx_y, test_lx_rx_y, steps=80)
example_input, example_target = next(iter(test_lx_rx_y.take(1)))

plt.figure(figsize=(6, 6))
#plt.subplot(1, 2, 1)
plt.imshow((example_target[0] + 1)/2, cmap='gray') # Right-view
plt.axis('off')
plt.savefig('test_model_1.png')

plt.figure(figsize=(6, 6))
#plt.subplot(1, 2, 2)
plt.imshow((generator(example_input, training=False)[0, ...] + 1)/2, cmap='gray') # Target
plt.axis('off')

plt.savefig('test_model_2.png')
