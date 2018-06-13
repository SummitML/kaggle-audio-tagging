SummitML Kaggle Audio Tagging
==============================

This is the group repository for [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging)
# Team Members
- rams

# Team Members
- Chang
- Carlos
- Rams

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


## Contributing

### Downloading the Repository

1. With [Git](https://git-scm.com/downloads) installed, clone this repository. If you're using the command line interface run...

  ```bash
  git clone https://github.com/SummitML/kaggle-audio-tagging.git
  ```

### Downloading the Kaggle Data

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

### Installing Dependencies

**Python**

There are various ways to install Python. Python3 is recommended for this project.

- [Python.org](https://www.python.org/downloads/)

- [Homebrew](https://docs.brew.sh/Homebrew-and-Python)

**Running the Virtual Environment**

With Python (and [PIP](https://pypi.org/)) installed, you're now ready to install a virtual environment and run it locally...

With [Pipenv](https://docs.pipenv.org/
)...

```bash
pip install pipenv
pipenv install
pipenv shell
```

*Note: When adding new dependencies to the project with Pipenv, remember to [generate](https://docs.pipenv.org/advanced/#generating-a-requirements-txt) an updated `requirements.txt` for teammates not using Pipenv*

With [Virtualenv](https://virtualenv.pypa.io/en/stable/)...

For Mac:
```bash
virtualenv venv # -> creates local environment
source venv/bin/activate # -> activates shell
pip install -r requirements.txt # -> dependencies
```

For Windows:
```bash
virtualenv venv # -> creates local environment
source venv/Scripts/activate # -> activates shell
pip install -r requirements.txt # -> dependencies
```


With [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)...

```bash
make create_environment
```

Once inside an isolated virtual environment (using any of the methods above), you can confirm your environment was set up correctly by running the provided `make` command...

```bash
make test_environment
```

### Git Workflow

Please reference Atlassians tutorial on the [Git Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).

**Branching Strategy**

For the purposes of this project, we'll rely on three primary branch types.

- Feature Branches - A working branch created by any team member for independent or paired development

- Develop - A reserved branch for code that has been peer reviewed and is production ready

- Master - A restricted branch that represents deployed production code
