import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

inp_data = pd.read_csv('/content/train.csv').fillna(' ')
X_train = inp_data['comment_text']
Y_train = train['toxic']

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(X_train)
X_train = word_vectorizer.transform(X_train)


X_train, X_test, Y_train, Y_test = train_test_split( X_train, Y_train, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(n_estimators=10000, max_leaf_nodes=500, random_state=21)
rfc.fit(X_train, Y_train)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score, precision_recall_fscore_support

predictions = rfc.predict(X_test)

print(classification_report(Y_test, predictions))
pd.DataFrame(confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])