import os
import zipfile
import pandas as pd

os.system('kaggle competitions download -c titanic -p ./data')

# Распаковка данных
with zipfile.ZipFile('./data/titanic.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

# Чтение данных
data = pd.read_csv('./data/train.csv')
data.to_csv('./data.csv', index=False)

print(data.head())
