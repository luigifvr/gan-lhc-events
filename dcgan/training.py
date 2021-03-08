import tensorflow as tf
import numpy as np
import readLHE
import invariants
import GAN
import hist_cb
import argparse
import time
import os
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description=('pp -> t tbar GAN training'))
parser.add_argument('file', help='path to LHE file')
parser.add_argument('-b', '--batch', default='100', type=int)
parser.add_argument('-i', '--iters', default='100', type=int)
parser.add_argument('-e', '--epochs', default='100', type=int)
parser.add_argument('-g', '--examples', default='10000', type=int)
parser.add_argument('-k', '--checkpoint', default='./training_checkpoints')
args = parser.parse_args()  

file = args.file
batch = args.batch
iters = args.iters
epochs = args.epochs
examples = args.examples
checkpoint = args.checkpoint

print('Training a DCGAN using events pp -> tt~ generated with Madgraph')
print('Training info:')
print('Batch size:', batch)
print('Number of iterations:', iters)
print('Number of epochs:', epochs)
print('Number of examples to generate:', examples)
print('Directory for checkpoints:', checkpoint)

def scaler(array):
    scal = MinMaxScaler(feature_range=(-1,1))
    scal.fit(array)
    return scal.transform(array)

def seedGen(to_gen, noise):
    while 1:
        yield tf.random.normal([to_gen, noise])
        
def train_step(batch_data):
    noise = tf.random.normal([batch, noise_size])
    h_noise = tf.random.normal([int(batch/2), noise_size])
    gen_data = generator(h_noise, training=True)
    d_data = np.concatenate((batch_data, gen_data))
    d_labels = np.concatenate((tf.ones(int(batch/2)), tf.zeros(int(batch/2))))
    
    discriminator.trainable = True
    discriminator.train_on_batch(gen_data, tf.zeros(int(batch/2)))
    discriminator.train_on_batch(batch_data, tf.ones(int(batch/2)))

    gan.train_on_batch(noise, tf.ones(int(batch)))

    d_loss_r, d_acc_r = discriminator.test_on_batch(batch_data, tf.ones(int(batch/2)))
    d_loss_g, d_acc_g = discriminator.test_on_batch(gen_data, tf.zeros(int(batch/2)))
    g_loss, g_acc = gan.test_on_batch(noise, tf.ones((batch)))	
    with open('./outputs/g_loss.txt','a') as g_loss_file, open('./outputs/d_loss.txt', 'a') as d_loss_file, open('./outputs/d_acc.txt', 'a') as d_acc_file :
           g_loss_file.write("{: >20} {: >20}".format(g_loss, g_acc))
           g_loss_file.write('\n')
           d_loss_file.write("{: >20} {:>20}".format(d_loss_r,d_loss_g))
           d_loss_file.write('\n')
           d_acc_file.write("{: >20} {:>20}".format(d_acc_r,d_acc_g))
           d_acc_file.write('\n')
        
evs = readLHE.readEvent(file)
init = readLHE.readInit(file)

invar = np.zeros((readLHE.NEvents(file),3))
for ev in evs:
    i=0
    invar[i,0] = invariants.GetEnergySquared(ev)
    invar[i,1] = invariants.GetMandelT(ev)
    invar[i,2] = invariants.GetRapidity(init, ev)
    i += 1

#invar[:,0] = (invar[:,0]-np.mean(invar[:,0]))/np.std(invar[:,0])
#invar[:,1] = (invar[:,1]-np.mean(invar[:,1]))/np.std(invar[:,1])
#invar[:,2] = (invar[:,2]-np.mean(invar[:,2]))/np.std(invar[:,2])


#invar[:,0] = scaler(invar[:,0])
#invar[:,1] = scaler(invar[:,1])

#invar = invar.reshape(-1, 2, 3, 1)

noise_size = 100 
generator = GAN.generatore(noise_size)
discriminator = GAN.discriminatore()
gan = GAN.gan_model(generator, discriminator)
ckpt = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

seeds = seedGen(examples, noise_size)
start = time.time()

#pretrain
#nse = tf.random.normal([len(invar), noise_size])
#gend = generator(nse, training=True)
#datas = np.concatenate((invar,gend))
#labs = np.concatenate((tf.ones(len(invar)), tf.zeros(len(invar))))
#discriminator.fit(datas, labs, epochs=1, batch_size=128)

#hist_cb.hist_callback(invar[:100000], generator.predict(next(seeds), batch_size=512), 0)
for epoch in range(epochs):
    for it in range(iters):
        row_shape = len(invar)
        rnd_idx = np.random.choice(row_shape, size=int(batch/2), replace=False)
        dts = invar[rnd_idx,:]
        train_step(dts)
    if (epoch+1) % 100 == 0 :
        ckpt.save(file_prefix = os.path.join(checkpoint, "ckpt"))
        print ('Epoch {} of {}. [{} sec]'.format(epoch, epochs, time.time()-start, ))
    #if (epoch+1) % 50 == 0:
        #hist_cb.hist_callback(invar[:100000], generator(next(seeds), batch_size=512), epoch) 

pred = generator.predict(next(seeds), batch_size=512)
np.savetxt('outputs/pred.txt', pred)

generator.save('outputs/generator.h5')
discriminator.save('outputs/discriminator.h5')
