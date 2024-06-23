import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

train = pd.read_csv('~/mlops/lab2/train.csv')

# Пример: обучение модели на основе случайного леса
X_train = train.drop(columns=['Survived'])
y_train = train['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
with open('~/mlops/lab2/model.pkl', 'wb') as file:
    pickle.dump(model, file)
