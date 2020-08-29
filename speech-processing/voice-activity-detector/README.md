### Setting up the environment:
Solution has been developed in Python 3.7 using anaconda distribution and TensorFlow 2.

Conda environment used to develop the solution can be recreated using the following command:

``` conda env create -f task4_env.yml ```

It can then be activated with:

``` conda activate vlad-speech ```

Note: it has been set up to run tensorflow on the CPU but ```tensorflow-gpu``` package is 
available  online which requires CUDA to be installed and set up. Running on GPU speeds up 
training greatly!

It is also possible to install required packages using pip with:

``` pip install -f requirements.txt ```

### Configurations:

First take a look at the `config.ini` file in Config folder. There are a few parameters that could
be adjusted. General parameters are set to the most optimal set up. If RAM is limited it is possible
to reduce the data size which will impact the performance but speed up the training.

It is also possible to set data location such that provided path points directly to the location of all the 
raw audio files and labels accordingly. For example default location would be `audio\` and directly within that folder 
all the audio files for all the data partitions stored. 
Keep models folder as is. It determines a relative path where models are stored.

Other files in the Config folder are parameters for the models such as layer sizes, learning rate and batch sizes.
`fnn_pars.json` parameters file will be used by default. One thing that could potentially be tuned is batch size as 
it is currently set to 512 and could be too high for target machine. If training is still too slow, `n_steps` could 
also be set to a lower number. 


### Running the solution:

Finally training can be run with the following command:

``` python task4_main.py train -v -s ```

where there are several arguments and flags to take note of. First argument is the mode to run in.
After training the mode could be set to `test` to evaluate the trained model.

`-v` flag stands for verbose and ensures that all the outputs are printed out along with plots.

`-s` flag is essential for running evaluation later as it determines if the model should be saved.

Once training has completed current model can be evaluated with:

``` python task4_main.py test -v ```




