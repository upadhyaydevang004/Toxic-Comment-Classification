import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


training=pd.read_csv('/content/drive/My Drive/train.csv')
testing=pd.read_csv('/content/drive/My Drive/test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = training[list_classes].values
list_sentences_train = training["comment_text"]
list_sentences_test = testing["comment_text"]
print(training.head())
X1_train = training.values[:,1]
Y_train = training.values[:,2]
print(Y_train)
print(X1_train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X1_train)
encoded = tokenizer.texts_to_sequences(X1_train)

print(X1_train[4])
print(encoded[8:9])


max_len = 0;
for i in range(0, len(encoded)):
    if max_len < len(encoded[i]):
        max_len = len(encoded[i])

max_len = max_len + 1
print(max_len)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( pad_sequences(encoded, maxlen=max_len, padding='post'), Y_train, test_size=0.2, random_state=42)


input_seq = Input(shape=(1404, ))


embed2 = LSTM(60, return_sequences=True,name='lstm_layer1')(Embedding(vocab_size, 100, input_length=max_len)(input_seq))
x1 = Dense(1, activation="sigmoid")(Dropout(0.1)(Dense(50, activation="relu")(GlobalMaxPool1D()(embed2))))
classifier = Model(input_seq, x1)
classifier.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(classifier.summary())
print(X_train)
print(X_test)

classifier.fit(X_train, Y_train, batch_size=512, epochs = 10, 
    verbose=1, validation_data= (X_test, Y_test))

predictions = classifier.predict(X_test)
print(predictions)

for i in range(len(predictions)):
    if predictions[i]<=0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1

print(predictions)

g=[]
for i in range(len(Y_test)):
  if(Y_test[i]==[1]):
    g.append(predictions[i])
print(g)

from sklearn.metrics import confusion_matrix, classification_report
Y_test = [[i] for i in Y_test]
print(Y_test)
print(classification_report(Y_test, predictions))

pd.DataFrame(
   confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr,_=roc_curve(Y_test,predictions)

print("roc_curve {}".format(roc_auc_score(Y_test,predictions)));

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()