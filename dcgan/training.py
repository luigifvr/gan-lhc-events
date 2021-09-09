import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import make_pipeline
import argparse
import time
import os
import pickle

import readLHE
import invariants
import GAN
import hist_cb

parser = argparse.ArgumentParser(description=('2 -> 2 GAN training'))
parser.add_argument('file', help='path to LHE file')
parser.add_argument('-b', '--batch', default='100', type=int)
parser.add_argument('-i', '--iters', default='1', type=int)
parser.add_argument('-e', '--epochs', default='100', type=int)
parser.add_argument('-g', '--examples', default='10000', type=int)
parser.add_argument('-p', '--path', default='./')
parser.add_argument('-v', '--verbose', default='500', type=int)
args = parser.parse_args()  

file = args.file
batch = args.batch
iters = args.iters
epochs = args.epochs
examples = args.examples
path = args.path
verbose = args.verbose

print('INFO: Training a DCGAN using events generated with Madgraph')
print('INFO: LHE file:', file)
print('INFO: Batch size:', batch)
print('INFO: Number of iterations:', iters)
print('INFO: Number of epochs:', epochs)
print('INFO: Number of examples to generate:', examples)
print('INFO: Outputs path:', path)

# Output file with info of training
def INFOoutput():
    with open(path+'outputs/trainingINFO.txt','a') as info_file:
        info_file.write("INFO: Input file name: {} \n".format(file))
        info_file.write("INFO: Batch size: {} \n".format(batch))
        info_file.write("INFO: Number of epochs: {} \n".format(epochs))
        info_file.write("INFO: Total time for training: {:.2f} \n".format(total_time))
        info_file.write("INFO: Preprocessing: \n")
        for i in range(len(pipeline)):
            info_file.write("		Scaler n. {} {} \n".format(i, pipeline[i]))
        
# Generator to get random noise of 'examples' size
def seedGen(to_gen, noise):
    while 1:
        yield tf.random.uniform([to_gen, noise])

# Train step        
def train_step(batch_data):
    noise = tf.random.uniform([batch, noise_size])
    gen_data = generator(noise, training=True)
    
    discriminator.trainable = True
    d_loss_g = discriminator.train_on_batch(gen_data, tf.zeros(int(batch)))
    d_loss_r = discriminator.train_on_batch(batch_data, tf.ones(int(batch)))
    d_loss = d_loss_g + d_loss_r
   
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, tf.ones(int(batch)))
    with open(path+'outputs/losses.txt','a') as loss_file:
           loss_file.write("{: >20} {: >20} \n".format(g_loss, d_loss))

# Create output if missing
if not os.path.exists(path):
	os.makedirs(path)
	os.makedirs(path+'outputs/')
	os.makedirs(path+'outputs/hist/')
	os.makedirs(path+'outputs/models/')
	os.makedirs(path+'outputs/preprocess/')

# Load and preprocess LHE file
evs = readLHE.readEvent(file)
init = readLHE.readInit(file)

invar = np.zeros((readLHE.NEvents(file),3))
i = 0
for ev in evs:
    invar[i,0] = invariants.GetEnergySquared(ev)
    invar[i,1] = invariants.GetMandelT(ev)
    invar[i,2] = invariants.GetRapidity(init, ev)
    i += 1

# Make pipeline PowerTransformer+MinMax
pipeline = make_pipeline(PowerTransformer(standardize=True), MinMaxScaler((-1,1)))
invar = pipeline.fit_transform(invar)

with open(path+'outputs/preprocess/pipeline.pickle', 'wb') as pip:
    pickle.dump(pipeline, pip)

#Define latent space dimension
noise_size = 100 

#Define models
generator = GAN.generatore(noise_size)
discriminator = GAN.discriminatore()
gan = GAN.gan_model(generator, discriminator)

seeds = seedGen(examples, noise_size)
start = time.time()

# Histograms callback
hist_cb.hist_callback(invar, generator.predict(next(seeds), batch_size=2048), 0, path=path)

# Training
for epoch in range(epochs):
    for it in range(iters):
        row_shape = len(invar)
        rnd_idx = np.random.choice(row_shape, size=int(batch), replace=False)
        dts = invar[rnd_idx,:]
        train_step(dts)
    if (epoch+1) % verbose == 0 :
        print ('INFO: Epoch {} of {}. [{} sec]'.format(epoch+1, epochs, time.time()-start, ))
        hist_cb.hist_callback(invar, generator.predict(next(seeds), batch_size=2048), epoch+1, path=path) 
        if epoch >= (epochs-verbose*50):
            generator.save(path+'outputs/models/generator_{}.h5'.format(epoch+1))

total_time = time.time()-start
discriminator.save(path+'outputs/models/discriminator.h5')

INFOoutput()
