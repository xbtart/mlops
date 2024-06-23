import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv('./test.csv')

# Загрузка модели
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)

X_test = test.drop(columns=['Survived'])
y_test = test['Survived']

# Оценка модели
y_pred = model.predict(X_test)

# Метрики
print(classification_report(y_test, y_pred))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
