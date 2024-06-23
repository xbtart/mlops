import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    train = pd.read_csv('train.csv')

    # Обучение модели
    X_train = train.drop(columns=['Survived'])
    y_train = train['Survived']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Сохранение модели
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    train_model()
