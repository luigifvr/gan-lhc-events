import tensorflow as tf
import numpy as np
import readLHE
import invariants
import GAN
import argparse
import time
import os

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

def train_step(batch_data):
    noise = tf.random.normal([batch, noise_size])
    h_noise = tf.random.normal([int(batch/2), noise_size])
    gen_data = generator(h_noise, training=True)
    d_data = np.concatenate((batch_data, gen_data))
    d_labels = np.concatenate((tf.ones(int(batch/2)), tf.zeros(int(batch/2))))

    discriminator.trainable = True 
    d_loss, d_acc = discriminator.train_on_batch(d_data, d_labels)
        
    g_loss = gan.train_on_batch(noise, tf.ones((batch)))
    
    with open('./outputs/g_loss.txt','a') as g_loss_file, open('./outputs/d_loss.txt', 'a') as d_loss_file, open('./outputs/d_acc.txt', 'a') as d_acc_file :
        g_loss_file.write(str(g_loss))
        g_loss_file.write('\n')
        d_loss_file.write(str(d_loss))
        d_loss_file.write('\n')
        d_acc_file.write(str(d_acc))
        d_acc_file.write('\n')

init, evs = readLHE.readEvent(file)

invar = np.zeros((len(evs),3))
for ev in range(len(evs)):
    invar[ev,0] = invariants.GetEnergySquared(evs[ev])
    invar[ev,1] = invariants.GetMandelT(evs[ev])
    invar[ev,2] = invariants.GetRapidity(init, evs[ev])

invar[:,0] = 2*(invar[:,0]-min(invar[:,0]))/(max(invar[:,0])-min(invar[:,0])) - 1
invar[:,1] = 2*(invar[:,1]-min(invar[:,1]))/(max(invar[:,1])-min(invar[:,1])) - 1
invar[:,2] = 2*(invar[:,2]-min(invar[:,2]))/(max(invar[:,2])-min(invar[:,2])) - 1

noise_size = 100 
generator = GAN.generatore(noise_size)
discriminator = GAN.discriminatore()
gan = GAN.gan_model(generator, discriminator)
ckpt = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

seeds = tf.random.normal([examples, noise_size])
start = time.time()
for epoch in range(epochs):
    for it in range(iters):
        row_shape = len(invar)
        rnd_idx = np.random.choice(row_shape, size=int(batch/2), replace=False)
        dts = invar[rnd_idx,:]
        train_step(dts)
    if (epoch+1) % 10 == 0 :
        ckpt.save(file_prefix = os.path.join(checkpoint, "ckpt"))
        print ('Epoch {} of {}. [{} sec]'.format(epoch, epochs, time.time()-start, ))

pred = generator(seeds, training=False)
np.savetxt('outputs/pred.txt', pred)
np.savetxt('outputs/seeds.txt', seeds)

generator.save('outputs/generator.h5')
discriminator.save('outputs/discriminator.h5')
