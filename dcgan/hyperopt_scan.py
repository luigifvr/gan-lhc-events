import yaml
import json
import argparse
import invariants
from readLHE import readEvent, readInit, NEvents
import hyperopt
from hyperopt import hp, fmin, STATUS_OK, Trials, tpe

import numpy as np
from dcgan import DCGAN
import scipy.special
import scipy.stats
import scipy.spatial

def get_kl_js(data_p, data_q):
    pdfp = scipy.stats.rv_histogram(np.histogram(data_p, bins=300))
    pdfq = scipy.stats.rv_histogram(np.histogram(data_q, bins=300))
    p = pdfp.pdf(np.linspace(min(data_p), max(data_p), 300))
    q = pdfq.pdf(np.linspace(min(data_q), max(data_q), 300))
    kl_div = scipy.special.kl_div(p,q)
    kl_div = kl_div[np.isfinite(kl_div)]
    js_div = scipy.spatial.distance.jensenshannon(p, q)**2
    return np.sum(kl_div), js_div

def hyperopt_training(dict_setup):
    data = build_and_process(dict_setup['file'])
    dcgan = DCGAN(dict_setup)
    dcgan.training(data, dict_setup['epochs'], dict_setup['examples'], batch_size=dict_setup['batch_size'])
    pred = dcgan.generate_events(dict_setup['examples'])
    loss = 0
    for i in range(3):
        loss += get_kl_js(data[:,i], pred[:,i])[0]
    ret = {'loss': loss/3, 'status': STATUS_OK}
    return ret

def build_and_process(file):
    evs = readEvent(file)
    init = readInit(file)
    invar = np.zeros((NEvents(file),3))
    i = 0
    for ev in evs:
        invar[i,0] = invariants.GetEnergySquared(ev)
        invar[i,1] = invariants.GetMandelT(ev)
        invar[i,2] = invariants.GetRapidity(init, ev)
        i += 1
    invar[:,0] = 2*(invar[:,0]-min(invar[:,0]))/(max(invar[:,0])-min(invar[:,0])) - 1
    invar[:,1] = 2*(invar[:,1]-min(invar[:,1]))/(max(invar[:,1])-min(invar[:,1])) - 1
    invar[:,2] = 2*(invar[:,2]-min(invar[:,2]))/(max(invar[:,2])-min(invar[:,2])) - 1 
    #add preprocessing
    return invar

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=('pp -> ttbar GAN training'))
    parser.add_argument('setup', help='path to setup file')
    parser.add_argument('-e', '--evals', default='100', type=int) 
    args = parser.parse_args()  
    setup_name = args.setup
    max_evals = args.evals

    setup_dict = yaml.load(open(setup_name), Loader=yaml.FullLoader)
    for key, val in setup_dict.items():
        if 'hp.' in str(val):
            setup_dict[key] = eval(val)
    
    trials = Trials()
    best = fmin(hyperopt_training, setup_dict, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    best_dict = hyperopt.space_eval(setup_dict, best)
    with open('hyperopt_best.json', 'w', encoding='utf8') as file:
        json.dump(best_dict, file)
