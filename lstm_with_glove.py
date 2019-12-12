import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import drive
from sklearn.model_selection import train_test_split
from keras.models import load_model

drive.mount('/content/drive',force_remount=True)

training=pd.read_csv('/content/drive/My Drive/train.csv')
testing=pd.read_csv('/content/drive/My Drive/test.csv')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = training[list_classes].values
list_sentences_train = training["comment_text"]
list_sentences_test = testing["comment_text"]
list_sentences_train[5]
data = training.values
X1_train = data[:,1]
Y_train = data[:,2]

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
vocab_size = len(tokenizer.word_index) + 1
X_train, X_test, Y_train, Y_test = train_test_split( pad_sequences(encoded, maxlen=max_len, padding='post'), Y_train, test_size=0.2, random_state=42)
embeddings_index = dict()
f = open('/content/drive/My Drive/glove.6B.100d.txt')
for line in f:
	embeddings_index[line.split()[0]] = np.asarray(line.split()[1:], dtype='float32')
f.close()
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
shape=Input(shape=(1404, ))
classifier = Model(shape, Dense(1, activation="sigmoid")(Dropout(0.1)(Dense(50, activation="relu")(GlobalMaxPool1D()(LSTM(60, return_sequences=True,name='lstm_layer1')(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=True)(shape)))))))
classifier.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(classifier.summary())

classifier.fit(X_train, Y_train, batch_size=512, epochs = 10, verbose=1, validation_data= (X_test, Y_test))
classifier.save('./GLOVE_CNN.h5')
classifier.save("./G_further.h5")
predictions = classifier.predict(X_test)
for i in range(len(predictions)):
    if predictions[i]<=0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1
Y_test = [[i] for i in Y_test]
print(Y_test)
print(classification_report(Y_test, predictions))
pd.DataFrame(
   confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])