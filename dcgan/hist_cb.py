import matplotlib.pyplot as plt
import scipy.special
import scipy.stats
import scipy.spatial
import numpy as np
import pickle

def hist_callback(true, predict, epoch, path='./'):
    with open(path+'outputs/preprocess/pipeline.pickle', 'rb') as pip:
        pipeline = pickle.load(pip)
    predict = predict.astype('float64')
    predict = pipeline.inverse_transform(predict)

    true = pipeline.inverse_transform(true)
    
    np.seterr(divide='ignore', invalid='ignore')
    bins = [np.logspace(np.log10(min(true[:,0])), np.log10(1.1e7), 50, endpoint=False), np.flip(-np.logspace(np.log10(-max(true[:,1])), np.log10(1.1e7),  50, endpoint=False)), np.arange(min(true[:,2]), max(true[:,2]), 0.1)]
    labels = ['s [GeV$^2$]', 't [GeV$^2$]', 'y']
    fig, ax = plt.subplots(3,3, figsize=(25,10), gridspec_kw={'height_ratios': [3,1,1]}, sharex='col')
    loc = ['upper right', 'upper left', 'upper right']
    xy = [[(0.78, 0.78),(0.78, 0.73)],[(0.02, 0.78),(0.02, 0.73)],[(0.78, 0.78),(0.78, 0.73)]]
    scale = ['log', 'log', 'linear']
    for i in range(3):
        valt, bs, _ = ax[0,i].hist(true[:,i], bins=bins[i], histtype='step', label='MadGraph', density=True)
        valg, bs, _ = ax[0,i].hist(predict[:,i], bins=bins[i], histtype='step', label='GAN', density=True)
        ax[0,i].set_xlabel(labels[i], fontsize=13)
        ax[0,i].legend(loc=loc[i])
        ax[0,i].annotate('KL-div = {:.2e}'.format(np.sum(get_kl_js(true[:,i], predict[:,i], bs)[0])), xy=xy[i][0], xycoords='axes fraction')
        ax[0,i].annotate('JS-div = {:.2e}'.format(get_kl_js(true[:,i], predict[:,i], bs)[1]), xy=xy[i][1], xycoords='axes fraction')
        ax[0,i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        ax[0,i].set_yscale(scale[i])
        ax[1,i].plot(bs[:-1], scipy.special.kl_div(valt,valg), linewidth='1')
        ax[1,i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        ax[1,i].set_ylabel('KL divergence')
        ax[2,i].step(bs[1:], valg/valt, where='pre')
        #ax[2,i].set_ylim(0.8, 1.2)
        ax[2,i].hlines(1, min(predict[:,i]), max(predict[:,i]), linestyle='dashed', color='black')
        ax[2,i].set_ylabel('GAN/MC')
        ax[2,i].fill_between(bs[1:], 1+1/np.sqrt(valt*(len(true[:,i])*np.diff(bs))), 1-1/np.sqrt(valt*(len(true[:,i])*np.diff(bs))), step='pre', alpha=0.2)
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.15,wspace=0.1)
    fig.suptitle('Distributions of MC and GAN events \n Epoch n. {}'.format(epoch))

    fig2, ax2 = plt.subplots(1,3, figsize=(25,6))
    for i in range(3):
        ax2[i].plot(predict[:,i], predict[:,(i+1)%3], 'o', markersize=1.3, label='GAN', color=u'#ff7f0e')
        ax2[i].plot(true[:,i%3], true[:,(i+1)%3],'o', markersize=1.3, label='MadGraph', color=u'#1f77b4')
        ax2[i].set_xlabel(labels[i%3], fontsize=13)
        ax2[i].set_ylabel(labels[(i+1)%3], fontsize=13)
        lgnd = ax2[i].legend(loc='best')
        ax2[i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        lgnd.legendHandles[0]._legmarker.set_markersize(6)
        lgnd.legendHandles[1]._legmarker.set_markersize(6)
    #fig2.tight_layout()
    fig2.suptitle('Correlation between invariants of MC and GAN events \n Epoch n. {}'.format(epoch))
    fig2.subplots_adjust(hspace=0.15,wspace=0.1) 
    plt.show(block=False)
    fig.savefig(path+'outputs/hist/hist_epoch_'+str(epoch)+'.png')
    fig2.savefig(path+'outputs/hist/corr_epoch_'+str(epoch)+'.png')
    plt.close(fig)
    plt.close(fig2)
    
def get_kl_js(data_p, data_q, bins):
    pdfp = scipy.stats.rv_histogram(np.histogram(data_p, bins=bins))
    pdfq = scipy.stats.rv_histogram(np.histogram(data_q, bins=bins))
    p = pdfp.pdf(np.linspace(min(data_p), max(data_p), 300))
    q = pdfq.pdf(np.linspace(min(data_q), max(data_q), 300))
    kl_div = scipy.special.kl_div(p,q)
    kl_div = kl_div[np.isfinite(kl_div)]
    js_div = scipy.spatial.distance.jensenshannon(p, q)**2
    return np.sum(kl_div), js_div
