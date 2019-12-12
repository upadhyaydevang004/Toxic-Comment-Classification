import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

class_names = ['toxic']

train = pd.read_csv('/content/gdrive/My Drive/train.csv').fillna(' ')
test = pd.read_csv('/content/gdrive/My Drive/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_features = word_vectorizer.transform(train_text)
test_features = word_vectorizer.transform(test_text)
train_target = train['toxic']
X_train, X_test, Y_train, Y_test = train_test_split( train_features, train_target, test_size=0.2, random_state=42)

"""**Logistic Regression**"""

classifier = LogisticRegression(C=0.1, solver='sag')
classifier.fit(X_train, Y_train)
predictions = classifier.predict_proba(X_test)[:, 1]
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score
print("roc_curve {}".format(roc_auc_score(Y_test,predictions)));
fpr, tpr,_=roc_curve(Y_test,predictions)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

for i in range(len(predictions)):
    if predictions[i]<=0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1

print("accuracy {}".format(accuracy_score(Y_test,predictions)));

print(classification_report(Y_test, predictions))
pd.DataFrame(confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])

"""**Naive Bayes**"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score, precision_recall_fscore_support
gnb = MultinomialNB()
gnb.fit(X_train, Y_train)
predictions = gnb.predict(X_test)
print("accuracy {}".format(accuracy_score(Y_test,predictions)))
print(classification_report(Y_test, predictions))
pd.DataFrame(confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])

"""**SVM**"""

from sklearn import svm
from sklearn.model_selection import GridSearchCV
svm_model = svm.SVC(C = 0.1, kernel = 'linear')
svm_model.fit(X_train, Y_train)
predictions = svm_model.predict(X_test)
print("accuracy {}".format(accuracy_score(Y_test,predictions)));
print(classification_report(Y_test, predictions))
pd.DataFrame(confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])