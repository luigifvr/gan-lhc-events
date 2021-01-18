import tensorflow as tf
from tensorflow.keras import layers

def generatore(noise_size) :
    model = tf.keras.Sequential()
    model.add(layers.Dense(200, input_shape=(noise_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((10,10,2)))
    assert model.output_shape == (None, 10, 10, 2)
    
    model.add(layers.Conv2DTranspose(64, kernel_size=2, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_normal'))
    assert model.output_shape == (None, 10, 10, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(32, kernel_size=2, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_normal'))
    assert model.output_shape == (None, 10, 10, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(16, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_normal'))
    assert model.output_shape == (None, 10, 10, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(3, activation='tanh'))
    
    return model

def discriminatore() :
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(200, use_bias=False, input_shape=(3,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 200)
    
    model.add(layers.Reshape((10,10,2)))
    assert model.output_shape == (None, 10, 10, 2)
    
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    assert model.output_shape == (None, 10, 10, 64)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    assert model.output_shape == (None, 10, 10, 32)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    assert model.output_shape == (None, 10, 10, 16)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

def gan_model(gen, disc):
    disc.trainable = False
    model = tf.keras.Sequential()
    model.add(gen)
    model.add(disc)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05), optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5))
    return model