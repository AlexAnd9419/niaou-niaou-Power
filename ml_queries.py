import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
import re
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

companies = ['Spartan', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green']
search_queries = ['Spatan', 'WD40', 'Astra-Tech', 'Speclum', 'Simple Red', 'Simple Green', 'SimpleGreen', 'WD40', 'Spartan Chemical', 'Spartan', 'Spectrum', 'Sectrum', 'WD35', 'Asta-Tech', 'Complex Green','WD10','Speculurtan','Spertan','Spartacus']
labels = ['Spartan','WD40','Asta-Tech','Spectrum','Simple Green','Simple Green','Simple Green','WD40','Spartan','Spartan','Spectrum','Spectrum','WD40','Asta-Tech','Simple Green','WD40','Spectrum','Spartan','Spartan']

data = pd.DataFrame({'query': search_queries, 'company': labels})

stemmer = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    return text

data['query'] = data['query'].apply(preprocess)
data['company'] = data['company'].apply(lambda x: x.lower())

X_train, X_test, y_train, y_test = train_test_split(data['query'], data['company'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on testing set: {accuracy}")

new_query = 'green cleaner'
new_query = preprocess(new_query)
new_query_vectorized = vectorizer.transform([new_query])

prediction = clf.predict(new_query_vectorized)

print(prediction)