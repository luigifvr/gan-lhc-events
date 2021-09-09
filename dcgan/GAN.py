import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import backend

def generatore(noise_size) :
    model = tf.keras.Sequential()
    model.add(layers.Dense(200, input_shape=(noise_size,), kernel_initializer='glorot_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((10,10,2)))
    assert model.output_shape == (None, 10, 10, 2)
    
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
   
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(16, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(3, activation='tanh'))

    return model

def discriminatore() :
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(200, use_bias=False, input_shape=(3,)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 200)
    
    model.add(layers.Reshape((10,10,2)))
    assert model.output_shape == (None, 10, 10, 2)
 
    model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 128)
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 64)
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 32)
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
    assert model.output_shape == (None, 10, 10, 16)
    #model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.LeakyReLU(alpha=0.2))  ##
    model.add(layers.Dropout(0.2))  ##

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adadelta(0.3))
    return model

def gan_model(gen, disc):
    disc.trainable = False
    model = tf.keras.Sequential()
    model.add(gen)
    model.add(disc)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adadelta(0.3))
    return model
