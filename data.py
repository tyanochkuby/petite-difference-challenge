import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



def get_data():
    raw_data = pd.read_csv(r'D:\hoojnia\py\UM\petite-difference-challenge\gender-classifier-DFE-791531.csv', encoding='MacRoman')

    y = raw_data['gender'].apply(
        lambda gender: 0 if gender == 'male' else 1
    )
    tweets = raw_data['text'].to_numpy()
    data = pd.DataFrame({
        'X': raw_data['text'],
        'y': raw_data['gender']
    })
    tfidf = TfidfVectorizer(
        ngram_range=(1,3),
        stop_words = 'english',
        max_features=2000
    )
    X = tfidf.fit_transform(tweets)
    print(f'X shape: {X.shape}')

    X = X.toarray() 
    y = y.to_numpy()
    return X, y
