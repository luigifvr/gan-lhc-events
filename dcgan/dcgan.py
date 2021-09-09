import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad

import time
import hist_cb
import numpy as np

def set_optimizer(opt, lr):
	if opt=='SGD':
		return SGD(lr)
	elif opt=='Adam':
		return Adam(lr)
	elif opt=='Adagrad':
		return Adagrad(lr)
	elif opt=='RMSprop':
		return RMSprop(lr)
	else:
		raise ExceptionError('Invalid optimizer')

class DCGAN():
    def __init__(self, dict_setup):
        self.noise_size = dict_setup['noise_size']
        optimizer = set_optimizer(dict_setup['optimizer'], dict_setup['learning_rate'])
        
        self.discriminator = self.make_discriminator(dict_setup['alpha'],
                                                     dict_setup['dropout'])
        self.discriminator.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        self.generator = self.make_generator(dict_setup['noise_size'], 
                                             dict_setup['alpha'], 
                                             dict_setup['activation'])
        
        self.gan = self.make_gan()
        self.gan.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    def make_discriminator(self, alpha=0.2, dropout=0.2):
        model = Sequential()
        
        model.add(Dense(200, use_bias=False, input_shape=(3,)))
        model.add(Reshape((10,10,2)))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
        model.add(LeakyReLU(alpha=alpha))
    
        model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
        model.add(LeakyReLU(alpha=alpha))
    
        model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
        model.add(LeakyReLU(alpha=alpha))
    
        model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform'))
        
        model.add(Flatten())
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout)) 

        model.add(Dense(1, activation='sigmoid'))
        return model
    
    def make_generator(self, noise_size=100, alpha=0.2, activation='tanh'):
        model = Sequential()
        
        model.add(Dense(200, input_shape=(noise_size,), kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))
        model.add(Reshape((10,10,2)))
    
        model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))
   
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))

        model.add(Conv2DTranspose(32, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))

        model.add(Conv2DTranspose(16, kernel_size=3, strides=1, padding=('same'), use_bias=False, kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))
    
        model.add(Flatten())
    
        model.add(Dense(3, activation=activation))
        return model
    
    def make_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model
    
    def training(self, data, epochs, examples, batch_size=1024, hist_callback=0):
        start = time.time()
        seeds = self.seedGen(examples)
        for epoch in range(epochs):
            row_shape = len(data)
            rnd_idx = np.random.choice(row_shape, size=int(batch_size), replace=False)
            batch_data = data[rnd_idx,:]
            noise = tf.random.uniform([batch_size, self.noise_size])
            gen_data = self.generator(noise, training=True)
            
            #Train discriminator
            self.discriminator.trainable = True
            self.discriminator.train_on_batch(gen_data, tf.zeros(int(batch_size)))
            self.discriminator.train_on_batch(batch_data, tf.ones(int(batch_size)))
            
            #train generator
            self.discriminator.trainable = False
            self.gan.train_on_batch(noise, tf.ones(int(batch_size)))
            
            #callbacks
            if (epoch+1) % verbose == 0 :
                print ('Epoch {} of {}. [{} sec]'.format(epoch+1, epochs, time.time()-start, ))
                hist_cb.hist_callback(data, self.generator.predict(next(seeds), batch_size=batch_size), epoch) 
                
    def load_model(self, gen_model, disc_model):
        self.generator = load_model(gen_model)
        self.discriminator = load_model(disc_model)
        
    def generate_events(self, examples, batch_size=512):
        seeds = self.seedGen(examples)
        pred = self.generator.predict(next(seeds), batch_size=batch_size)
        return pred
        
    def seedGen(self, to_gen):
        while 1:
            yield tf.random.normal([to_gen, self.noise_size])
