import numpy as np
import pandas as pd

from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.layers import Conv1D, GlobalMaxPool1D, Input
from keras.layers import Dense, Dropout, MaxPooling1D
from keras.layers import Flatten
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


data = pd.read_csv('/content/train.csv')

data = data.values
X_train = data[:,1]
Y_train = data[:,2]

print(X_train[1])
print(Y_train[1])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

print(X_train.shape)
print(Y_train.shape)

input_seq = Input(shape=(10000, ))
x = Dense(512, activation='relu')(input_seq)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
classi = Dense(1, activation='sigmoid')(x)
model = Model(input_seq, classi)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, batch_size=512, epochs = 30, 
    verbose=1, validation_data= (X_test, Y_test))

predictions = model.predict(X_test)

for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print(predictions)


Y_test = [[i] for i in Y_test]

print(classification_report(Y_test, predictions))

pd.DataFrame(
   confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])

predictions = model.predict(X_test)


fpr, tpr,_=roc_curve(Y_test,predictions)

print("roc_curve {}".format(roc_auc_score(Y_test,predictions)));


plt.figure()

plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()



for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0


print(classification_report(Y_test, predictions))
pd.DataFrame(confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])

data = pd.read_csv('/content/test.csv')

data = data.values
X_test = data[:,1]

X_test = word_vectorizer.transform(X_test)

predictions = model.predict(X_test)

for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print(predictions)