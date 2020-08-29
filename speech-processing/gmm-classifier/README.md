### Set-up:
Solution has been developed in Python 3.7.

Packages used are listed in requirements.txt file and can be installed with pip via:

`pip install -r requirements.txt`

If using anaconda it is possible to recreate the environment like this:

`conda env create -f task7_env.yml`

Then activate environment with:

`conda activate vlad-speech`

### Data Location:
Data for vowels is supplied with the submission but due to large file size data for 
speakers has been omitted. Speaker data should be stored in `speakers` folder with all
the `.dat` files located directly in that folder without any subdirectories.

### Running the Solution:
Finally the code for both vowel and speaker classification can be run:
`python speechtech_gmm.py`