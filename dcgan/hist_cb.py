import matplotlib.pyplot as plt
import scipy.special
import scipy.stats
import scipy.spatial
import numpy as np

def hist_callback(true, predict, epoch):
    bins = [300, 300, 300]
    xs = ['s', 't', 'y']
    xlims = [(-1.01, 1.01), (-1.01, 1.01), (-1.01, 1.01)]
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    fig.suptitle('Epoch n. {}'.format(epoch))
    for i in range(3):
        ax[i].hist(true[:,i], bins=bins[i], histtype='step', density=True, label='MadGraph')
        ax[i].hist(predict[:,i], bins=bins[i], histtype='step', density=True, label='GAN')
        ax[i].annotate('KL = {:.2e}'.format(get_kl_js(true[:,i], predict[:,i])[0]), xy=(0.65, 0.84), xycoords='axes fraction')
        ax[i].annotate('JS = {:.2e}'.format(get_kl_js(true[:,i], predict[:,i])[1]), xy=(0.65, 0.80), xycoords='axes fraction')
        ax[i].set_xlim(xlims[i])
        ax[i].set_xlabel(xs[i])
        ax[i].legend(loc='upper right')
    plt.show(block=False)
    fig.savefig('outputs/hist/hist_epoch_'+str(epoch)+'.png')
    plt.close(fig)
        
        
def get_kl_js(data_p, data_q):
    pdfp = scipy.stats.rv_histogram(np.histogram(data_p, bins=300))
    pdfq = scipy.stats.rv_histogram(np.histogram(data_q, bins=300))
    p = pdfp.pdf(np.linspace(min(data_p), max(data_p), 300))
    q = pdfq.pdf(np.linspace(min(data_q), max(data_q), 300))
    kl_div = scipy.special.kl_div(p,q)
    kl_div = kl_div[np.isfinite(kl_div)]
    js_div = scipy.spatial.distance.jensenshannon(p, q)**2
    return np.sum(kl_div), js_div
