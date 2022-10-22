from pathlib import Path
import os
import zipfile
import numpy as np
import pandas as pd
import logisticregression
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# create a folder in the project directory and download to it as a zip file
api.competition_download_files('titanic', 'data_titanic')
# example : https://www.kaggle.com/c/titanic

with zipfile.ZipFile('data_titanic/titanic.zip', 'r') as zipref:
    zipref.extractall('data_titanic/')

train = pd.read_csv('data_titanic/train.csv')
print(train.columns)
print(train)
