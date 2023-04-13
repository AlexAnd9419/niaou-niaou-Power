import re

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

companies = ['Spartan', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green']
search_queries = ['Spatan', 'WD40', 'Astra-Tech', 'Speclum', 'Simple Red', 'Simple Green', 'SimpleGreen', 'WD40', 'Spartan Chemical', 'Spartan', 'Spectrum', 'Sectrum', 'WD35', 'Asta-Tech', 'Complex Green', 'WD10', 'Speculurtan', 'Spertan', 'Spartacus', 'Sperian', 'Astro Tech', 'Simpel Green', 'W40', 'Asta Tech', 'Spec Trum', 'Sparyan', 'WD-40', 'Spectrum Chemicals', 'Simpul Green', 'Spertun', 'WD50', 'Sparten', 'Simpl Green', 'AstroTech', 'Specrtrum', 'Spartan Chemicals', 'Astro-Tech', 'Spectrum Chemical', 'Spectrum Technology', 'Spectro-Tech', 'Spectrum Tech', 'Spectrum Chemical Corp', 'Spectrum Inc', 'Spectrum Corp', 'Spectrum Science', 'Spectra-Tech', 'Spectron Tech', 'Spectrum Technologies', 'Spectrum Products', 'Spectrum Analytical', 'Spectrum Dynamics', 'Spectra-Tech Inc', 'Spectramed', 'Spectrum Sciences', 'Spectrum Engineering', 'Spectrulite', 'Spectra Green', 'Spectrum Diagnostics', 'Spectrum Industries', 'Spectra-Tec', 'Spectrus', 'SpectraMotive', 'Spectrum Microbiology', 'Spectro-Science', 'Spectra Group', 'Spectrachem', 'SpectraMat','Sparten', 'Spatran', 'Spartun', 'Spartin', 'Sparton', 'Sparta', 'Spatan', 'Sparten', 'Spartane', 'Spartenium', 'Spartum', 'Spartian', 'Spartenix', 'SpartanX', 'Spartanium', 'Spardan', 'Sparthian', 'Spartann', 'Spartand', 'Spartanite','Asta Tech', 'Astar Tech', 'Asta Techs', 'Asta-Tek', 'Astro Tech', 'Asto-Tech', 'Asti-Tech', 'Astr-Tech', 'Ast-Tech', 'Astech', 'Asta-Techs', 'AstaTex', 'AstaTexh', 'AstaTechs', 'Asta-Tach', 'Asta-Teh', 'Asta-Teech', 'Asta-Techinc', 'Asta-Tec', 'Asta-Techo','Simple Greene', 'Simle Green', 'Simple Geen', 'Simpe Green', 'Simple Greeno', 'Simpl Green', 'Simple Greenz', 'Simple Gren', 'Simple Grn', 'Simple Geren', 'Simple Greenie', 'Simple Grean', 'Simple Grien', 'Simple Grren', 'Simle Grene', 'Simpe Greene', 'Simplle Green', 'Simpli Green', 'Simpl Green', 'Sympal Green','WD-40', 'W40', 'WD fourty', 'WD 40s', 'W D 40', 'WD F0', 'WD 4O', 'W4D0', 'WDO4', 'WD4o', 'Wd40s', 'Wd-40', 'Wd 40', 'W-D 40', 'W40D', 'wD40', 'Wd forty', 'WD04', 'WD40s', 'Wd forty','Spatan', 'WD40', 'Astra-Tech', 'Speclum', 'Simple Red', 'Simple Green', 'SimpleGreen', 'WD40', 'Spartan Chemical', 'Spartan', 'Spectrum', 'Sectrum', 'WD35', 'Asta-Tech', 'Complex Green', 'WD10', 'Speculurtan', 'Spertan', 'Spartacus', 'Sperian', 'Astro Tech', 'Simpel Green', 'W40', 'Asta Tech', 'Spec Trum', 'Sparyan', 'WD-40', 'Spectrum Chemicals', 'Simpul Green', 'Spertun', 'WD50', 'Sparten', 'Simpl Green', 'AstroTech', 'Specrtrum', 'Spartan Chemicals', 'Astro-Tech', 'Spectrum Chemical', 'Spectrum Technology', 'Spectro-Tech', 'Spectrum Tech', 'Spectrum Chemical Corp', 'Spectrum Inc', 'Spectrum Corp', 'Spectrum Science', 'Spectra-Tech', 'Spectron Tech', 'Spectrum Technologies', 'Spectrum Products', 'Spectrum Analytical', 'Spectrum Dynamics', 'Spectra-Tech Inc', 'Spectramed', 'Spectrum Sciences', 'Spectrum Engineering', 'Spectrulite', 'Spectra Green', 'Spectrum Diagnostics', 'Spectrum Industries', 'Spectra-Tec', 'Spectrus', 'SpectraMotive', 'Spectrum Microbiology', 'Spectro-Science', 'Spectra Group', 'Spectrachem', 'SpectraMat','Sparten', 'Spatran', 'Spartun', 'Spartin', 'Sparton', 'Sparta', 'Spatan', 'Sparten', 'Spartane', 'Spartenium', 'Spartum', 'Spartian', 'Spartenix', 'SpartanX', 'Spartanium', 'Spardan', 'Sparthian', 'Spartann', 'Spartand', 'Spartanite','Asta Tech', 'Astar Tech', 'Asta Techs', 'Asta-Tek', 'Astro Tech', 'Asto-Tech', 'Asti-Tech', 'Astr-Tech', 'Ast-Tech', 'Astech', 'Asta-Techs', 'AstaTex', 'AstaTexh', 'AstaTechs', 'Asta-Tach', 'Asta-Teh', 'Asta-Teech', 'Asta-Techinc', 'Asta-Tec', 'Asta-Techo','Simple Greene', 'Simle Green', 'Simple Geen', 'Simpe Green', 'Simple Greeno', 'Simpl Green', 'Simple Greenz', 'Simple Gren', 'Simple Grn', 'Simple Geren', 'Simple Greenie', 'Simple Grean', 'Simple Grien', 'Simple Grren', 'Simle Grene', 'Simpe Greene', 'Simplle Green', 'Simpli Green', 'Simpl Green', 'Sympal Green','WD-40', 'W40', 'WD fourty', 'WD 40s', 'W D 40', 'WD F0', 'WD 4O', 'W4D0', 'WDO4', 'WD4o', 'Wd40s', 'Wd-40', 'Wd 40', 'W-D 40', 'W40D', 'wD40', 'Wd forty', 'WD04', 'WD40s', 'Wd forty']

labels = ['Spartan', 'WD40', 'Asta-Tech', 'Spectrum', 'Simple Green', 'Simple Green', 'Simple Green', 'WD40', 'Spartan', 'Spartan', 'Spectrum', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green', 'WD40', 'Spectrum', 'Spartan', 'Spartan', 'Spartan', 'Asta-Tech', 'Simple Green', 'WD40', 'Asta-Tech', 'Spectrum','Spartan', 'WD40', 'Spectrum', 'Simple Green', 'Spartan', 'WD40', 'Spartan', 'Simple Green', 'Asta-Tech', 'Spectrum','Spartan', 'Asta-Tech', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum','Simple Green','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','Spartan', 'WD40', 'Asta-Tech', 'Spectrum', 'Simple Green', 'Simple Green', 'Simple Green', 'WD40', 'Spartan', 'Spartan', 'Spectrum', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green', 'WD40', 'Spectrum', 'Spartan', 'Spartan', 'Spartan', 'Asta-Tech', 'Simple Green', 'WD40', 'Asta-Tech', 'Spectrum','Spartan', 'WD40', 'Spectrum', 'Simple Green', 'Spartan', 'WD40', 'Spartan', 'Simple Green', 'Asta-Tech', 'Spectrum','Spartan', 'Asta-Tech', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum','Simple Green','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40']

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
print(data)


X_train, X_test, y_train, y_test = train_test_split(data['query'], data['company'], test_size=0.2, random_state=9)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

clf = SGDClassifier(loss='log_loss', penalty='l2',alpha=0.001)

batch_size = 80
n_epochs = 5

for epoch in range(n_epochs):
    n_batches = X_train.shape[0] // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
        clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Epoch {epoch+1}: Test accuracy = {accuracy:.3f}")

y_prob = clf.predict_proba(X_test)


y_pred = clf.predict(X_test)
manufacturer_pred = [companies[x] for x in y_pred]
print(manufacturer_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on testing set: {accuracy}")

new_query = 'WD40'
new_query = preprocess(new_query)
print(new_query)
new_query_vectorized = vectorizer.transform([new_query])
print(new_query_vectorized)
probs = clf.predict_proba(new_query_vectorized)
print(probs)
prediction = clf.predict(new_query_vectorized)
prediction_name = [companies[x] for x in prediction]
print(prediction_name,prediction)