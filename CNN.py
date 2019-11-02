import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalMaxPool1D, Input
from keras.layers import Dense, Dropout, MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv("/content/drive/My Drive/ALDA/train.csv")

data = data.values
X_train = data[:,1]
Y_train = data[:,2]

print(Y_train)
print(X_train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
encoded = tokenizer.texts_to_sequences(X_train)

max_len = 0;
for i in range(0, len(encoded)):
    if max_len < len(encoded[i]):
        max_len = len(encoded[i])

max_len = max_len + 1
print(max_len)
vocabulary_size = len(tokenizer.word_index) + 1
print(vocabulary_size)

padded_inp = pad_sequences(encoded, maxlen=max_len, padding='post')
print(padded_inp)

X_train, X_test, Y_train, Y_test = train_test_split( padded_inp, Y_train, test_size=0.2, random_state=42)

max_len = 0;
for i in range(0, len(X_train)):
    if max_len < len(X_train[i]):
        max_len = len(X_train[i])
        
print(max_len)
print(X_train.shape)

inp = Input(shape=(max_len, ))
embed = Embedding(vocabulary_size, 100, input_length=max_len)(inp)
classifier = Model(inp, Dense(1, activation='sigmoid')(Dense(128, activation='relu')(GlobalMaxPool1D()(Conv1D(128, 4, activation='relu')(MaxPooling1D(2, 1)(Conv1D(256, 4 , activation='relu')(embed)))))))
classifier.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(classifier.summary())

classifier.fit(X_train, Y_train, batch_size=512, epochs = 20, verbose=1, validation_data= (X_test, Y_test))

predictions = classifier.predict(X_test)

for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0


Y_test = [[i] for i in Y_test]

print(classification_report(Y_test, predictions))

pd.DataFrame(
   confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])