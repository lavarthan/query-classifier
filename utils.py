import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('data/preprocessed.csv')
data = data[['type', 'category_id', 'news_porter_stemmed']]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))

features = tfidf.fit_transform(data.news_porter_stemmed.astype('U')).toarray()
labels = data.category_id

data.columns = ['newstype', 'category_id', 'news_porter_stemmed']

category_id_df = data[['newstype', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'newstype']].values)

with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)


# function for classification
def text_classifier(query):
    text_features = tfidf.transform([query])
    predictions = model.predict_proba(text_features)
    x = np.matrix(predictions)
    y = x.tolist()
    if max(y[0]) > 0.4:
        print(max(y[0]))
        return id_to_category[y[0].index(max(y[0]))]
    else:
        print(max(y[0]))
        return 'No class found'
