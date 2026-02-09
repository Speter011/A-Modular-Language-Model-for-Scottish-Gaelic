This repository contains my initial tests using the spaCy library to create a Part of Speech tagger for Scottish Gaelic using the ARCOSG (Annotated Reference Corpus of Scottish Gaelic) database.

Several scripts have been implemented to make training, testing, and packaging models easier and sami-automated. This will make future testing much quicker.

>**Note! All below scripts must be run from the `scripts` folder to avoid path issues.**

1. `convert_to_spacy.py` must be ran first. This converts the raw data to the `.spacy` format. It also implements some standardisation to the tokenization and some changes are made to the tagmap (`data/raw/gd-parole.map`) as well for better tag accuracy. Change directories at the beggining of the file to test different datasets.
2. `make_config.py` is next, it creates the basic set up spaCy needs to run training and testing.
3. `train_model.py` uses the above two resultant files to train and save models. Trained models can be found in `training/output/` in the `model-best` and `model-last` folders.
4. `evaluate.py` loads the best model and creates and `evaluation_metrics.json` in the `results` folder using the test set of the data. This details all accuracy, precision, recall, f-score, etc. for all posssible pipelines.
5. `package.py` creates a new packaged model out of the best one in the `training/output/` folder. Its name and version number must be modified through the `package.py` rather than the meta.json as running the code overwrites it. Installation and usage info is outputted after packaging is successful and may be copy-pasted for ease of use in the terminal.

Alternatively command line might be used to train, ecaluate and package models using the `data/raw` folder as the corpus.
[spaCy usage guide](https://spacy.io/usage/training)