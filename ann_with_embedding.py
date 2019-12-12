import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.layers import Conv1D, GlobalMaxPool1D, Input
from keras.layers import Dense, Dropout, MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

inp_data = pd.read_csv('/content/train.csv')
inp_data = inp_data.values
X_train = inp_data[:,1]
Y_train = inp_data[:,2]
print(X_train[3])
print(Y_train[3])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
encodedTrain = tokenizer.texts_to_sequences(X_train)

print(X_train[1])
print(encodedTrain[1])

maxLen = 0;
for i in range(0, len(encodedTrain)):
    if maxLen < len(encodedTrain[i]):
        maxLen = len(encodedTrain[i])

maxLen = maxLen + 1
vocabSize = len(tokenizer.word_index) + 1
print(maxLen)
print(vocabSize)

padded_data = pad_sequences(encodedTrain, maxlen=maxLen, padding='post')

print(padded_data)
X_train, X_test, Y_train, Y_test = train_test_split( padded_data, Y_train, test_size=0.2, random_state=42)

input_seq = Input(shape=(maxLen, ))
embedding_Layer = Embedding(vocabSize, 100, input_length=maxLen)(input_seq)
x = Flatten()(embedding_Layer)
x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
classify = Dense(1, activation='sigmoid')(x)
model = Model(input_seq, classify)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, batch_size=512, epochs = 40, 
    verbose=1, validation_data= (X_test, Y_test))

predictions = model.predict(X_test)

for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print(predictions)

from sklearn.metrics import confusion_matrix, classification_report

Y_test = [[i] for i in Y_test]

print(classification_report(Y_test, predictions))

pd.DataFrame(
   confusion_matrix(Y_test, predictions),
   index = [['Actual', 'Actual'], ['Not Toxic', 'Toxic']],
   columns = [['Predicted', 'Predicted'], ['Not Toxic', 'Toxic']])

predictions = model.predict(X_test)
print(predictions)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr,_=roc_curve(Y_test,predictions)

print("roc_curve {}".format(roc_auc_score(Y_test,predictions)));


plt.figure()

plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score

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

encoded = tokenizer.texts_to_sequences(X_test)
padded_docs = pad_sequences(encoded, maxlen=maxLen, padding='post')

predictions = model.predict(padded_docs)

for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print(predictions)