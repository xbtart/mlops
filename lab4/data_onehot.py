import pandas as pd

def onehot_data():
    df = pd.read_csv('data/titanic_filled.csv')
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df.to_csv('data/titanic_onehot.csv', index=False)

if __name__ == '__main__':
    onehot_data()
