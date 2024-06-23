import pandas as pd

def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df.to_csv('data/titanic.csv', index=False)
    
if __name__ == '__main__':
    load_data()
