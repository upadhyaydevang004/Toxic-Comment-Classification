import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalMaxPool1D, Input
from keras.layers import Dense, Dropout, MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('/content/drive/My Drive/train.csv').values
X_train = data[:,1]
Y_train = data[:,2]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
encoded = tokenizer.texts_to_sequences(X_train)
print(X_train[4])
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
shape = Input(shape=(1404, ))
classifier = Model(shape, Dense(1, activation='sigmoid')(Dense(128, activation='relu')(Dense(512, activation='relu')(GlobalMaxPool1D()(Conv1D(128, 4, activation='relu')(MaxPooling1D(2, 1)(Conv1D(256, 4, activation='relu')(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=True)(shape)))))))))
classifier.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(classifier.summary())
classifier.fit(X_train, Y_train, batch_size=512, epochs = 30, 
    verbose=1, validation_data= (X_test, Y_test))
predictions = classifier.predict(X_test)
for i in range(len(predictions)):
    if predictions[i]<=0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1

Y_test = [[i] for i in Y_test]
print(classification_report(Y_test, predictions))

pd.DataFrame(
   confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])