import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data():
    data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

    # Удаление ненужных столбцов и заполнение пропусков
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Заполнение пропусков только в числовых столбцах
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Преобразование категориальных данных
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    # Нормализация данных
    scaler = StandardScaler()
    data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

    # Разделение данных на тренировочный и тестовый наборы
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)

if __name__ == "__main__":
    prepare_data()
