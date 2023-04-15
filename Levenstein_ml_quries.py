import pandas as pd
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
import numpy as np
import Levenshtein

companies = ['Spartan', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green']
search_queries = ['Spatan', 'WD40', 'Astra-Tech', 'Speclum', 'Simple Red', 'Simple Green', 'SimpleGreen', 'WD40', 'Spartan Chemical', 'Spartan', 'Spectrum', 'Sectrum', 'WD35', 'Asta-Tech', 'Complex Green', 'WD10', 'Speculurtan', 'Spertan', 'Spartacus', 'Sperian', 'Astro Tech', 'Simpel Green', 'W40', 'Asta Tech', 'Spec Trum', 'Sparyan', 'WD-40', 'Spectrum Chemicals', 'Simpul Green', 'Spertun', 'WD50', 'Sparten', 'Simpl Green', 'AstroTech', 'Specrtrum', 'Spartan Chemicals', 'Astro-Tech', 'Spectrum Chemical', 'Spectrum Technology', 'Spectro-Tech', 'Spectrum Tech', 'Spectrum Chemical Corp', 'Spectrum Inc', 'Spectrum Corp', 'Spectrum Science', 'Spectra-Tech', 'Spectron Tech', 'Spectrum Technologies', 'Spectrum Products', 'Spectrum Analytical', 'Spectrum Dynamics', 'Spectra-Tech Inc', 'Spectramed', 'Spectrum Sciences', 'Spectrum Engineering', 'Spectrulite', 'Spectra Green', 'Spectrum Diagnostics', 'Spectrum Industries', 'Spectra-Tec', 'Spectrus', 'SpectraMotive', 'Spectrum Microbiology', 'Spectro-Science', 'Spectra Group', 'Spectrachem', 'SpectraMat','Sparten', 'Spatran', 'Spartun', 'Spartin', 'Sparton', 'Sparta', 'Spatan', 'Sparten', 'Spartane', 'Spartenium', 'Spartum', 'Spartian', 'Spartenix', 'SpartanX', 'Spartanium', 'Spardan', 'Sparthian', 'Spartann', 'Spartand', 'Spartanite','Asta Tech', 'Astar Tech', 'Asta Techs', 'Asta-Tek', 'Astro Tech', 'Asto-Tech', 'Asti-Tech', 'Astr-Tech', 'Ast-Tech', 'Astech', 'Asta-Techs', 'AstaTex', 'AstaTexh', 'AstaTechs', 'Asta-Tach', 'Asta-Teh', 'Asta-Teech', 'Asta-Techinc', 'Asta-Tec', 'Asta-Techo','Simple Greene', 'Simle Green', 'Simple Geen', 'Simpe Green', 'Simple Greeno', 'Simpl Green', 'Simple Greenz', 'Simple Gren', 'Simple Grn', 'Simple Geren', 'Simple Greenie', 'Simple Grean', 'Simple Grien', 'Simple Grren', 'Simle Grene', 'Simpe Greene', 'Simplle Green', 'Simpli Green', 'Simpl Green', 'Sympal Green','WD-40', 'W40', 'WD fourty', 'WD 40s', 'W D 40', 'WD F0', 'WD 4O', 'W4D0', 'WDO4', 'WD4o', 'Wd40s', 'Wd-40', 'Wd 40', 'W-D 40', 'W40D', 'wD40', 'Wd forty', 'WD04', 'WD40s', 'Wd forty','Spatan', 'WD40', 'Astra-Tech', 'Speclum', 'Simple Red', 'Simple Green', 'SimpleGreen', 'WD40', 'Spartan Chemical', 'Spartan', 'Spectrum', 'Sectrum', 'WD35', 'Asta-Tech', 'Complex Green', 'WD10', 'Speculurtan', 'Spertan', 'Spartacus', 'Sperian', 'Astro Tech', 'Simpel Green', 'W40', 'Asta Tech', 'Spec Trum', 'Sparyan', 'WD-40', 'Spectrum Chemicals', 'Simpul Green', 'Spertun', 'WD50', 'Sparten', 'Simpl Green', 'AstroTech', 'Specrtrum', 'Spartan Chemicals', 'Astro-Tech', 'Spectrum Chemical', 'Spectrum Technology', 'Spectro-Tech', 'Spectrum Tech', 'Spectrum Chemical Corp', 'Spectrum Inc', 'Spectrum Corp', 'Spectrum Science', 'Spectra-Tech', 'Spectron Tech', 'Spectrum Technologies', 'Spectrum Products', 'Spectrum Analytical', 'Spectrum Dynamics', 'Spectra-Tech Inc', 'Spectramed', 'Spectrum Sciences', 'Spectrum Engineering', 'Spectrulite', 'Spectra Green', 'Spectrum Diagnostics', 'Spectrum Industries', 'Spectra-Tec', 'Spectrus', 'SpectraMotive', 'Spectrum Microbiology', 'Spectro-Science', 'Spectra Group', 'Spectrachem', 'SpectraMat','Sparten', 'Spatran', 'Spartun', 'Spartin', 'Sparton', 'Sparta', 'Spatan', 'Sparten', 'Spartane', 'Spartenium', 'Spartum', 'Spartian', 'Spartenix', 'SpartanX', 'Spartanium', 'Spardan', 'Sparthian', 'Spartann', 'Spartand', 'Spartanite','Asta Tech', 'Astar Tech', 'Asta Techs', 'Asta-Tek', 'Astro Tech', 'Asto-Tech', 'Asti-Tech', 'Astr-Tech', 'Ast-Tech', 'Astech', 'Asta-Techs', 'AstaTex', 'AstaTexh', 'AstaTechs', 'Asta-Tach', 'Asta-Teh', 'Asta-Teech', 'Asta-Techinc', 'Asta-Tec', 'Asta-Techo','Simple Greene', 'Simle Green', 'Simple Geen', 'Simpe Green', 'Simple Greeno', 'Simpl Green', 'Simple Greenz', 'Simple Gren', 'Simple Grn', 'Simple Geren', 'Simple Greenie', 'Simple Grean', 'Simple Grien', 'Simple Grren', 'Simle Grene', 'Simpe Greene', 'Simplle Green', 'Simpli Green', 'Simpl Green', 'Sympal Green','WD-40', 'W40', 'WD fourty', 'WD 40s', 'W D 40', 'WD F0', 'WD 4O', 'W4D0', 'WDO4', 'WD4o', 'Wd40s', 'Wd-40', 'Wd 40', 'W-D 40', 'W40D', 'wD40', 'Wd forty', 'WD04', 'WD40s', 'Wd forty']


labels = ['Spartan', 'WD40', 'Asta-Tech', 'Spectrum', 'Simple Green', 'Simple Green', 'Simple Green', 'WD40', 'Spartan', 'Spartan', 'Spectrum', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green', 'WD40', 'Spectrum', 'Spartan', 'Spartan', 'Spartan', 'Asta-Tech', 'Simple Green', 'WD40', 'Asta-Tech', 'Spectrum','Spartan', 'WD40', 'Spectrum', 'Simple Green', 'Spartan', 'WD40', 'Spartan', 'Simple Green', 'Asta-Tech', 'Spectrum','Spartan', 'Asta-Tech', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum','Simple Green','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','Spartan', 'WD40', 'Asta-Tech', 'Spectrum', 'Simple Green', 'Simple Green', 'Simple Green', 'WD40', 'Spartan', 'Spartan', 'Spectrum', 'Spectrum', 'WD40', 'Asta-Tech', 'Simple Green', 'WD40', 'Spectrum', 'Spartan', 'Spartan', 'Spartan', 'Asta-Tech', 'Simple Green', 'WD40', 'Asta-Tech', 'Spectrum','Spartan', 'WD40', 'Spectrum', 'Simple Green', 'Spartan', 'WD40', 'Spartan', 'Simple Green', 'Asta-Tech', 'Spectrum','Spartan', 'Asta-Tech', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum', 'Spectrum','Simple Green','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spectrum','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Spartan','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Asta-Tech','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','Simple Green','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40','WD40']


data = pd.DataFrame({'query': search_queries, 'company': labels})
p_intercept=[]
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

# Calculate Levenshtein distance between query and label
data['levenshtein_dist'] = data.apply(lambda row: Levenshtein.distance(row['query'], row['company']), axis=1)

X_train, X_test, y_train, y_test = train_test_split(data[['query', 'levenshtein_dist']], data['company'], test_size=0.1, random_state=9)
print(X_test)

# Use CountVectorizer to convert text data to numerical data
vectorizer = CountVectorizer()
X_train_text = vectorizer.fit_transform(X_train['query'])
X_test_text = vectorizer.transform(X_test['query'])

# Combine text data and Levenshtein distance
X_train = np.hstack((X_train_text.toarray(), np.array(X_train['levenshtein_dist']).reshape(-1,1)))
X_test = np.hstack((X_test_text.toarray(), np.array(X_test['levenshtein_dist']).reshape(-1,1)))

clf = LogisticRegression(penalty='elasticnet',C=0.2,l1_ratio=0.5,solver='saga',max_iter=10000)
fitted_val=clf.fit(X_train, y_train)
for inter in clf.intercept_:
    p_intercept.append((math.exp(inter))/(1+math.exp(inter)))
print(clf.intercept_,p_intercept)

y_prob = clf.predict_proba(X_test)

y_pred = clf.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on testing set: {accuracy}")

# new_query = 'orange Green'
# new_query = preprocess(new_query)
# new_query_vectorized = vectorizer.transform([new_query])
#
# prediction = clf.predict(new_query_vectorized)
# print(clf.predict_proba(new_query_vectorized))
# print(prediction)