from scipy.spatial.distance import jensenshannon
from scipy.stats import rv_histogram, norm
from scipy.special import kl_div
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, DivergingNorm
import seaborn as sns

import readLHE
import invariants

class Event():
    def __init__(self, particles):
        self.particles = particles
        for p in particles:
            p.event = self

class Particle():
    def __init__(self, val_labels):
        for key, val in val_labels.items():
            setattr(self, key, val)
        self.p = [self.e, np.sqrt(self.px**2 + self.py**2), self.pz]

def gen_sample(model, pipe_file, n_ev):
    gan_gen = tf.keras.models.load_model(model)
    noise = tf.random.uniform([int(n_ev), 100])
    gan_pred = gan_gen.predict(noise, batch_size=4096*8)
    with open(pipe_file, 'rb') as pipe:
        pipeline = pickle.load(pipe)
    gan_pred = gan_pred.astype('float64')
    gan_pred = pipeline.inverse_transform(gan_pred)
    return gan_pred

def get_kl_js(data_p, data_q, bins):
    pdfp = rv_histogram(np.histogram(data_p, bins=bins))
    pdfq = rv_histogram(np.histogram(data_q, bins=bins))
    p = pdfp.pdf(bins)
    q = pdfq.pdf(bins)
    kldv = kl_div(p,q)
    kldv[kldv == inf] = 0
    js_div = jensenshannon(p, q)**2
    return np.sum(kldv), js_div

def load_sample(file):
   evs = readLHE.readEvent(file)
   init = readLHE.readInit(file)
   invar = np.zeros((readLHE.NEvents(file),3))
   i = 0
   for ev in evs:
      invar[i,0] = invariants.GetEnergySquared(ev)
      invar[i,1] = invariants.GetMandelT(ev)
      invar[i,2] = invariants.GetRapidity(init, ev)
      i += 1
   return invar

def get_event(s, t, y, beamE=6500, m=173.):
    E3_cm = np.sqrt(s)/2
    x_2 = np.sqrt(s)/2/beamE*np.exp(-y)
    x_1 = x_2*np.exp(2*y)
    cos_theta_cm = 1 + 2*t/s
    sin_theta_cm = np.sqrt(1-cos_theta_cm**2)
    pz_cm = np.sqrt(s)/2*cos_theta_cm
    pz = pz_cm*np.cosh(y)+E3_cm*np.sinh(y)
    E3 = E3_cm*np.cosh(y)+pz_cm*np.sinh(y)
    pt = np.sqrt(E3**2-(pz**2 + m**2))
    E1 = x_1*beamE
    E2 = x_2*beamE
    d_p1 = {'e': E1, 'px': 0., 'py': 0., 'pz': E1}
    d_p2 = {'e': E2, 'px': 0., 'py': 0., 'pz': -E2}
    d_p3 = {'e': E3, 'px': pt, 'py': 0., 'pz': pz}
    d_p4 = {'e': E1+E2-E3, 'px': -pt, 'py': 0., 'pz': E1-E2-pz}
    p1 = Particle(d_p1)
    p2 = Particle(d_p2)
    p3 = Particle(d_p3)
    p4 = Particle(d_p4)
    ev = Event([p1,p2,p3,p4])
    return ev

def plt_inv(slice1, slice2, train_sample):
    np.seterr(divide="ignore", invalid="ignore")
    labels = ['s [GeV$^2$]', 't [GeV$^2$]', 'y']
    fig, ax = plt.subplots(3,3, figsize=(25,10), gridspec_kw={'height_ratios': [3,1,1]}, sharex='col')
    loc = ['upper right', 'upper left', 'upper right']
    xy = [[(0.78, 0.78),(0.78, 0.73)],[(0.02, 0.78),(0.02, 0.73)],[(0.78, 0.78),(0.78, 0.73)]]
    xscale = ['symlog', 'symlog', 'linear']
    yscale = ['log', 'log', 'linear']
    bins = [np.logspace(np.log10(min(slice1[:,0])), np.log10(0.6e7), 50, endpoint=False), np.flip(-np.logspace(np.log10(-max(slice1[:,1])), np.log10(0.6e7),  50, endpoint=False)), np.arange(min(slice1[:,2]), max(slice1[:,2]), 0.1)]
    for i in range(3): 
        valt, bs, _ = ax[0,i].hist(slice1[:,i], bins=bins[i], histtype='step', label='MadGraph', density=True)
        valg, bs, _ = ax[0,i].hist(slice2[:,i], bins=bins[i], histtype='step', label='GAN', density=True)
        ax[0,i].set_xlabel(labels[i], fontsize=13)
        ax[0,i].legend(loc=loc[i])
        ax[0,i].annotate('KL-div = {:.2e}'.format(np.sum(get_kl_js(slice1[:,i], slice2[:,i], bs)[0])), xy=xy[i][0], xycoords='axes fraction')
        ax[0,i].annotate('JS-div = {:.2e}'.format(get_kl_js(slice1[:,i], slice2[:,i], bs)[1]), xy=xy[i][1], xycoords='axes fraction')
        ax[0,i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        ax[0,i].set_yscale(yscale[i])
        ax[0,i].set_xscale(xscale[i])
        ax[1,i].plot(bs[:-1], kl_div(valt,valg), linewidth='1')
        ax[1,i].set_ylabel('KL divergence')
        ax[2,i].step(bs[1:], valg/valt, where='pre', color=u'#ff7f0e')
        ax[2,i].set_ylim(0.8, 1.2)
        ax[2,i].hlines(1, min(slice2[:,i]), max(slice2[:,i]), color=u'#1f77b4')
        ax[2,i].set_ylabel('GAN/MC')
        ax[2,i].fill_between(bs[1:], 1+1/np.sqrt(valt*(len(slice1[:,i])*np.diff(bs))), 1-1/np.sqrt(valt*(len(slice1[:,i])*np.diff(bs))), step='pre', alpha=0.2)
        ratio_err = 1-valg/valt
        mad_err = 1/np.sqrt(valt*len(slice1)*np.diff(bs))
        si = ratio_err/mad_err
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.15,wspace=0.1)
    fig.suptitle('Distributions of MC and GAN events \n Training sample: {} events, Number of events: {}'.format(train_sample, len(slice1)))

def plt_corr(slice1, slice2):
    labels = ['s [GeV$^2$]', 't [GeV$^2$]', 'y']
    scalecov = ['log', 'symlog', 'linear']
    fig, ax = plt.subplots(1,3, figsize=(25,6))
    for i in range(3):
       ax[i].plot(slice2[:,i], slice2[:,(i+1)%3], 'o', markersize=1.3, label='GAN', color=u'#ff7f0e')
       ax[i].plot(slice1[:,i%3], slice1[:,(i+1)%3],'o', markersize=1.3, label='MadGraph', color=u'#1f77b4')
       ax[i].set_xlabel(labels[i%3], fontsize=13)
       ax[i].set_ylabel(labels[(i+1)%3], fontsize=13)
       lgnd = ax[i].legend(loc='best')
       ax[i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
       lgnd.legendHandles[0]._legmarker.set_markersize(6)
       lgnd.legendHandles[1]._legmarker.set_markersize(6)
       ax[i].set_xscale(scalecov[i])
       ax[i].set_yscale(scalecov[(i+1)%3])
    fig.suptitle('Correlation between invariants of MC and GAN events', fontsize=15)

def plt_err(slice1, slice2):
    bins = [np.logspace(np.log10(min(slice1[:,0])), np.log10(1e7), 50, endpoint=False), np.flip(-np.logspace(np.log10(-max(slice1[:,1])), np.log10(1e7),  50, endpoint=False)), np.arange(min(slice1[:,2]), max(slice1[:,2]), 0.1)]
    labels = ['s [GeV$^2$]', 't [GeV$^2$]', 'y']
    scalecov = ['log', 'symlog', 'linear']
    fig1 = plt.figure(figsize=(25,15))
    fig2, ax = plt.subplots(1,3, figsize=(25,6))
    for i in range(3):
        lims = [(1e5, max(slice1[:,0])), (min(slice1[:,1]), max(slice1[:,1])), (min(slice1[:,2]),max(slice1[:,2]))]
        ax1 = fig1.add_axes([np.linspace(0,1,3, endpoint=False)[i], 1, 0.30, 0.30])
        ax2 = fig1.add_axes([np.linspace(0,1,3, endpoint=False)[i], 1, 0.24, 0.30], frameon=False)
        h_mg, x_bin, y_bin = np.histogram2d(slice1[:,i], slice1[:,(i+1)%3], bins=[bins[i],bins[(i+1)%3]])
        h_gan, x_bin, y_bin = np.histogram2d(slice2[:,i], slice2[:,(i+1)%3], bins=[bins[i], bins[(i+1)%3]])
        ax2.set_xlabel(labels[i%3], fontsize=13)
        ax2.set_ylabel(labels[(i+1)%3], fontsize=13)
        rat_err = 1-h_gan/h_mg
        mg_err = 1/np.sqrt(h_mg)
        np.nan_to_num(rat_err/mg_err, copy=False, nan=0, posinf=0, neginf=0)
        img = ax1.imshow((rat_err/mg_err).T, cmap='RdBu_r', aspect='auto', origin='lower', norm=DivergingNorm(vmin=-5, vcenter=0, vmax=5))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xscale(scalecov[i])
        ax2.set_yscale(scalecov[(i+1)%3])
        ax2.set_xlim(lims[i][0], lims[i][1])
        ax2.set_ylim(lims[(i+1)%3][0], lims[(i+1)%3][1])
        bar = fig1.colorbar(img, ax=ax1)
        bar.set_label('Standard deviations\nof the ratio between MG and GAN', fontsize=13)
        ax[i].hist((rat_err/mg_err).flatten(), histtype='step', bins=np.arange(-10,11,1), density=True, label=labels[i])
        ratio = (rat_err/mg_err).flatten()
        ratio = ratio[~np.isnan(ratio)]
        (mu, sigma) = norm.fit(ratio)
        ax[i].annotate('Mean: {:.2f}\nSigma: {:.2f}'.format(mu, sigma), xy=(0.78, 0.78), xycoords='axes fraction', fontsize=13)
        ax[i].legend()
    fig2.text(0.5, -0.001,'Density ditribution of the standard deviations', fontsize=15, ha='center', va='center')

