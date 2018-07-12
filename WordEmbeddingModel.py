# Christian Lahr
# 07.07.2018

import pandas as pd
import numpy as np
import tqdm
tqdm.tqdm.pandas()

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelEncoder

asset_path = r"AZD RNN/Assets"
MODEL_PATH = r"Neustart/models/WordEmbeddungDense1"
train = pd.read_pickle(asset_path + "/train_data_firstPage.pickle")
eval_test = pd.read_pickle(asset_path + "/eval_data_firstPage.pickle")

# bad label at index 58374!!
train = train.drop([58374])
# train = train.iloc[:100]
# eval = eval.iloc[:100]

# transforme labels into integers
labelEncoder = LabelEncoder()
labelEncoder.fit(np.hstack([train.Label, eval_test.Label]))
train["Label_int"] = labelEncoder.transform(train.Label)
eval_test["Label_int"] = labelEncoder.transform(eval_test.Label)
#labelEncoder.inverse_transform(lab)

print("Create one hot vector")
vocab_size = 10000
train_encoded = [one_hot(d, vocab_size) for d in train.Text]
eval_test_encoded = [one_hot(d, vocab_size) for d in eval_test.Text]

max_length = 200
print("Pad sequence to max length", max_length)
train_padded = pad_sequences(train_encoded, maxlen=max_length, padding='post')
eval_test_padded = pad_sequences(eval_test_encoded, maxlen=max_length, padding='post')

model = Sequential()
model.add(Embedding(vocab_size, 150, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = Adam(lr = 0.001, clipnorm = 10), loss = binary_crossentropy, metrics=['acc'])
print(model.summary())

# test eval split
middle = int(np.round(len(eval_test_padded) / 2,0))
eval_padded = eval_test_padded[:middle,:]
eval = eval_test.iloc[:middle]
test_padded = eval_test_padded[middle:,:]
test = eval_test.iloc[middle:]

print("Start training")
batch_size = 64
checkpoint = ModelCheckpoint(MODEL_PATH + 'model.hd5f',save_best_only=True,verbose = True)
early_stop = EarlyStopping(patience=5)
model.fit(train_padded, train.Label_int.values, batch_size=batch_size, epochs=15, verbose=0, validation_data = (eval_padded, eval.Label_int.values), callbacks=[checkpoint, early_stop])
print("Start test")
loss, accuracy = model.evaluate(test_padded, test.Label_int.values, verbose=0)
print('Accuracy: %f' % (accuracy*100))
