# GAN model for event generation
### Deep convolutional model for *pp*→*tt̅*:
Start training using:

`
./dcgan/trainng.py events_file -e epochs
`

Output will be saved in:

`
./outputs/
`

Possible options are:
- `-e` number of epochs
- `-b` dimension of the batch
- `-i` number of iterations
- `-g` number of examples to generate
- `-c` directory used to save checkpoint

### Hyperopt scan of hyperparameters:
Run an optimization of hyperparameters with hyperopt with:
'
python3 hyperopt_scan.py <dict_setup.yaml>
'

options:
- '-e' max number of evaluations for hyperopt
