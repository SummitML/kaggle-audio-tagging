SummitML Kaggle Audio Tagging
==============================

This is the group repository for [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging)

## Contributing

##### Downloading the Repository

1. With [Git](https://git-scm.com/downloads) installed, clone this repository.

  If you're using the command line interface...

  ```bash
  git clone https://github.com/SummitML/kaggle-audio-tagging.git
  ```

##### Downloading the Kaggle Data

1. Create a directory at the root of your project called **data**

  ```bash
  cd /path/to/kaggle-audio-tagging/
  mkdir data
  ```

1. Download project data files from [Kaggle](https://www.kaggle.com/c/freesound-audio-tagging/data)

1. Find your downloaded files and move them into your newly created `data` directory.

  ```bash
  mv /path/to/downloaded/file /path/to/kaggle-audio-tagging/data
  ```

##### Installing Dependencies

**Python**

There are various ways to install Python. Python3 is recommended for this project.

- [Python.org](https://www.python.org/downloads/)

- [Homebrew](https://docs.brew.sh/Homebrew-and-Python)


With Python installed, run your virtual environment locally...

```bash
pip install virtualenv
```

```bash
virtualenv venv
source venv/bin/activate
```
To install required dependencies, run...


```bash
make requirements
```




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
