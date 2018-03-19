
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')
y_pred = pd.read_csv('sample_submission.csv')
train_df.isnull().any(),test_df.isnull().any()
np.random.seed(19)
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] 
cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_rows =train_df.shape[0]
train_df.drop(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)

data = pd.concat([train_df, test_df])
del train_df
del test_df

#If you will have errors with nltk, add 
# nltk.download()
#and choose useful libraries

stop_words = set(stopwords.words('english'))

def preprocess_input(comment):
# remove the extra spaces at the end.
    comment = comment.strip()
# lowercase to avoid difference between 'hate', 'HaTe'
    comment = comment.lower()
# remove the escape sequences. 
    comment = re.sub('[\s0-9]',' ', comment)
# Use nltk's word tokenizer to split the sentence into words. 
    words = nltk.word_tokenize(comment)
# removing the commonly used words.
    words = [word for word in words if len(word) > 2]
    comment = ' '.join(words)
    return comment

data.comment_text = data.comment_text.apply(lambda row: preprocess_input(row))

test = data[train_rows:]
train = data[:train_rows]
del data

for c in cols:
    clf = LogisticRegression()
    clf.fit(train, y[c])
    y_pred[c] = clf.predict_proba(test)[:,1]

y_pred.to_csv('submission.csv', index=False)
