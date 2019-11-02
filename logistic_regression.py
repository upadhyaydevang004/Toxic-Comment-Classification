import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score

class_names = ['toxic']

train = pd.read_csv('/content/drive/My Drive/ALDA/train.csv').fillna(' ')
test = pd.read_csv('/content/drive/My Drive/ALDA/test.csv').fillna(' ')

train_data = train['comment_text']
test_data = test['comment_text']
all_data = pd.concat([train_data, test_data])

word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',stop_words='english',
    ngram_range=(1, 1), max_features=10000)
word_vectorizer.fit(all_data)
train_features = word_vectorizer.transform(train_data)
test_features = word_vectorizer.transform(test_data)
X_train, X_test, Y_train, Y_test = train_test_split( train_features, train['toxic'], test_size=0.2, random_state=42)

model = LogisticRegression(C=1, solver='sag')
model.fit(X_train, Y_train)
predictions = model.predict_proba(X_test)[:, 1]


print("roc_curve {}".format(roc_auc_score(Y_test,predictions)));

fpr, tpr,_=roc_curve(Y_test,predictions)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
plt.savefig('LR')

for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0


print("accuracy {}".format(accuracy_score(Y_test,predictions)));

print(classification_report(Y_test, predictions))
pd.DataFrame(confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])


