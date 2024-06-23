import pandas as pd

def fillna_data():
    df = pd.read_csv('data/titanic_modified.csv')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df.to_csv('data/titanic_filled.csv', index=False)

if __name__ == '__main__':
    fillna_data()
