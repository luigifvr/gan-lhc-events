import numpy as np
import tensorflow as tf
import os

batch = 32
buffer = 10000

epoche = 100
noise_dim = 100
num_examples_to_generate = 10000

seed = tf.random.normal([num_examples_to_generate, noise_dim])
np.savetxt('seeds.txt', seed)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
