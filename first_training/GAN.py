import tensorflow as tf
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from variables import *

def generatore() :
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7, 7, 256)
    
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding=('same'), use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding=('same'), use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding=('same'), activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    
    model.add(layers.Flatten())
    model.add(layers.Dense(3))
    
    return model

def discriminatore() :
    model = tf.keras.Sequential()
    model.add(layers.Dense(28*28*1, use_bias=False, input_shape=(3,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 28*28*1)
    
    model.add(layers.Reshape((28,28,1)))
    assert model.output_shape == (None, 28, 28, 1)
    
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same'))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    assert model.output_shape == (None, 7*7*128)
    model.add(layers.Dense(1))
    
    return model

generator = generatore()
discriminator = discriminatore()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                 discriminator_optimizer=discriminator_opt,
                                 generator=generator,
                                 discriminator=discriminator)

def discriminator_loss(real_output, fake_output) :
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output) :
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train_step(img):
    global batch, noise_dim
    noise = tf.random.normal([batch, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :
        gen_img = generator(noise, training=True)
        
        real_output = discriminator(img, training=True)
        fake_output = discriminator(gen_img, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        with open('loss_gen.txt','a') as loss_gen_file, open('loss_disc.txt', 'a') as loss_disc_file :
            loss_gen_file.write(str(gen_loss.numpy()))
            loss_gen_file.write('\n')
            loss_disc_file.write(str(disc_loss.numpy()))
            loss_disc_file.write('\n')

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def train(dataset, epoche, step=0) :
    
    for epoch in range(epoche-step):
        start = time.time()
        
        for img_batch in dataset :
            train_step(img_batch)
        
        if (epoch+1) % 5 == 0 :
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print ('Time for epoch {} is {} sec'.format(epoch + step, time.time()-start, ))
