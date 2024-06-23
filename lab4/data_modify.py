import pandas as pd

def modify_data():
    df = pd.read_csv('data/titanic.csv')
    df = df[['Pclass', 'Sex', 'Age']]
    df.to_csv('data/titanic_modified.csv', index=False)

if __name__ == '__main__':
    modify_data()

