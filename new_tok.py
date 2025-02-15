
import pickle
from keras.preprocessing.text import Tokenizer
import pandas as pd

data = pd.read_csv('url_data.csv')
X = data['url']

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)