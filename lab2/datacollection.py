import os
import zipfile
import pandas as pd

os.system('kaggle competitions download -c titanic -p ~/mlops/lab2/data')

# Распаковка данных
with zipfile.ZipFile('~/mlops/lab2/data/titanic.zip', 'r') as zip_ref:
    zip_ref.extractall('~/mlops/lab2/data')

# Чтение данных
data = pd.read_csv('~/mlops/lab2/data/train.csv')
data.to_csv('~/mlops/lab2/data.csv', index=False)

print(data.head())
