[![TF](https://img.shields.io/badge/TensorFlow-v2.4.1-orange.svg)](https://shields.io/)
[![MG5](https://img.shields.io/badge/MG5_@NLO-v2.9.2-blue.svg)](https://shields.io/)
[![Python](https://img.shields.io/badge/Python-v3.7-brightgreen.svg)](https://shields.io/)

# GAN LHC 2→2 events
This is a repo for a generative LHC events model. The output of the network is a tuple which define the event in terms of Mandelstam invariants **s**, **t**, and the pseudorapidity in the parton scattering reference frame **y**.

These results are tied to a Master's thesis project and the taken steps for its developing are collected in notebooks stored in `./dcgan/notebooks`.
1. Initial results ✔️
2. Hyperopt results ✔️
3. Channels results ✖️

### Training of the deep convolutional GAN:
To start the training, initially generate a LHE file using MadGraph5. Then, it is possible to call `training.py` to train the model. To transform the `.lhe` file into an array, the script uses `read_LHE.py` to save the particles of each event and then `invariants.py` to calculate the input features.

The progress of the training procedure is checked by `hist_cb.py` which produces the histograms of the input features accordingly to the verbosity parameter.
#### Start training:
`
python3 ./dcgan/trainng.py file.lhe <OPTIONS>
`

The available options are:
- `-e` number of epochs;
- `-b` dimension of the input batch;
- `-i` number of iterations;
- `-g` number of examples to generate;
- `-p` directory used to save the output files;
- `-v` verbose frequency.

### Hyperopt scan of training parameters:
The same training procedure has been implemented in a class format in ``dcgan.py`` in order to perform an hyperopt scan through `hyperopt_scan.py`. The file needed is a dictionary which contains the variable parameters defined accordingly to the Hyperopt documentation followed by the number of evaluations. An example setup is presented in `example_setup_hyperopt.yaml`. 
#### Start Hyperopt scan:
`
python3 hyperopt_scan.py dict_setup.yaml -e <#ofEvals>
`

options:
- `-e` max number of evaluations for Hyperopt
