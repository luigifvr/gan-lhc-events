import tensorflow as tf
import numpy as np
import os
from variables import *
import readLHE
import invariants
import GAN
from GAN import *

file = "unweighted_events.lhe"
init, evs = readLHE.readEvent(file)

invar = np.zeros((len(evs),3))
for ev in range(len(evs)):
    invar[ev,0] = invariants.GetEnergySquared(evs[ev])
    invar[ev,1] = invariants.GetMandelT(evs[ev])
    invar[ev,2] = invariants.GetRapidity(init, evs[ev])

invar = invar.reshape(invar.shape[0], 3).astype('float32')
invar[:,0] = invar[:,0]/max(abs(invar[:,0]))
invar[:,1] = invar[:,1]/max(abs(invar[:,1]))
invar[:,2] = invar[:,2]/max(abs(invar[:,2]))
invar_dataset = tf.data.Dataset.from_tensor_slices(invar).shuffle(buffer).batch(batch)

GAN.train(invar_dataset, epoche)
pred = generator(seed, training=False)
np.savetxt('pred.txt', pred)

generator.save('generator.h5')
discriminator.save('discriminator.h5')
