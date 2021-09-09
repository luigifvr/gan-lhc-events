import yaml
import json
import argparse
import invariants
import numpy as np

import hyperopt
from hyperopt import hp, fmin, STATUS_ok, Trials, tpe
import scipy.special
import scipy.stats
import scipy.spatial
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

import invariants
from readLHE import readEvent, readInit, NEvents
from dcgan import DCGAN

def get_kl_js(data_p, data_q, bins):
    pdfp = scipy.stats.rv_histogram(np.histogram(data_p, bins=bins))
    pdfq = scipy.stats.rv_histogram(np.histogram(data_q, bins=bins))
    p = pdfp.pdf(bins)
    q = pdfq.pdf(bins)
    kl_div = scipy.special.kl_div(p,q)
    kl_div[kl_div == inf] = 0
    js_div = scipy.spatial.distance.jensenshannon(p, q)**2
    return np.sum(kl_div), js_div

def hyperopt_training(dict_setup):
    data = build_and_process(dict_setup['file'])
    dcgan = DCGAN(dict_setup)
    dcgan.training(data, dict_setup['epochs'], dict_setup['examples'], batch_size=dict_setup['batch_size'])
    pred = dcgan.generate_events(dict_setup['examples'])
    loss = 0
    for i in range(3):
        loss += get_kl_js(data[:,i], pred[:,i], dict_setup['bins'])[0]
    ret = {'loss': loss/3, 'status': STATUS_OK}
    return ret

def build_and_process(setup):
    evs = readEvent(setup["file"])
    init = readInit(setup["file"])
    invar = np.zeros((NEvents(setup["file"]),3))
    i = 0
    for ev in evs:
        invar[i,0] = invariants.GetEnergySquared(ev)
        invar[i,1] = invariants.GetMandelT(ev)
        invar[i,2] = invariants.GetRapidity(init, ev)
        i += 1
    if setup["preprocessing"] == "std":
        pipeline = make_pipeline(StandardScaler())
        invar = pipeline.fit_transform(invar)
    if setup["preprocessing"] == "minmax":
        pipeline = make_pipeline(PowerTransformer(Standardize=True), MinMaxScaler((-1,1)))
        invar = pipeline.fit_transform(invar)

    #save pipeline
    with open(setup["output"]+'pipeline.pickle', 'wb') as pip:
        pickle.dump(pipeline, pip) 
    return invar

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=('2 -> 2 GAN training'))
    parser.add_argument('setup', help='path to setup file')
    parser.add_argument('-e', '--evals', default='100', type=int) 
    args = parser.parse_args()  
    setup_name = args.setup
    max_evals = args.evals

    print("INFO: Starting Hyperopt run: \n")
    print("INFO: Using SETUP file: {} \n".format(setup_name))
    print("INFO: Number of evaluations: {} \n".format(max_evals))

    setup_dict = yaml.load(open(setup_name), Loader=yaml.FullLoader)
    for key, val in setup_dict.items():
        if 'hp.' in str(val):
            setup_dict[key] = eval(val)
    
    trials = Trials()
    best = fmin(hyperopt_training, setup_dict, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    best_dict = hyperopt.space_eval(setup_dict, best)
    with open('hyperopt_best.json', 'w', encoding='utf8') as file:
        json.dump(best_dict, file)
