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
