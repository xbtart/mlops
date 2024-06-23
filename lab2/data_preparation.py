import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('~/mlops/lab2/data.csv')

# Удаление ненужных столбцов и заполнение пропусков
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data = data.fillna(data.mean())

# Преобразование категориальных данных
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Нормализация данных
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Разделение данных на тренировочный и тестовый наборы
train, test = train_test_split(data, test_size=0.2, random_state=42)

train.to_csv('~/mlops/lab2/train.csv', index=False)
test.to_csv('~/mlops/lab2/test.csv', index=False)
