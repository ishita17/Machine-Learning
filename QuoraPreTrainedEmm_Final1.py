from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras import preprocessing 
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
import matplotlib.pyplot as plt


train_df = pd.read_csv(r'C:\Users\ishit\QuoraData\train.csv', low_memory = False)
test_df = pd.read_csv(r'C:\Users\ishit\QuoraData\test.csv', low_memory = False)

train_df.head()
questions = train_df['question_text']
values = train_df['target']

questions = list(questions)
values = list(values)

#Tokenizing the text
max_len = 50
training_samples = 600000
validation_samples = 400000
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen = max_len)

values = np.asarray(values)
print('shape of data tensor:' , data.shape)
print('shape of label tensor:', values.shape)

x_train = data[:training_samples]
y_train = data[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = data[training_samples: training_samples + validation_samples]

#Preparing Glove file
def readGloveFile(gloveFile):
    with open(gloveFile, encoding = 'utf8') as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token 

        for line in f:
            record = line.strip().split()
            token = record[0] # take the token (word) from the text line
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) # associate the Glove embedding vector to a that token (word)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
            wordToIndex[tok] = kerasIdx # associate an index to a token (word)
            indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove

# Create Pretrained Keras Embedding Weights Matrix
def createPretrainedEmbeddingMatrix(wordToGlove, wordToIndex):
    vocabLen = len(wordToIndex) + 1  # adding 1 to account for masking
    embDim = next(iter(wordToGlove.values())).shape[0]  # works with any glove dimensions (e.g. 50)

    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        embeddingMatrix[index, :] = wordToGlove[word] # create embedding: word index to Glove word embedding

    return vocabLen, embDim, embeddingMatrix


max_len = 50
wordToIndex, indexToWord, wordToGlove = readGloveFile(r'C:\Users\ishit\glove.6B.100d.txt')
vocabLen, embDim, embeddingMatrix = createPretrainedEmbeddingMatrix(wordToGlove, wordToIndex)


model = Sequential()
model.add(Embedding(vocabLen,embDim, weights=[embeddingMatrix], input_length= max_len, trainable = False))
model.add(LSTM(embDim, input_shape= (None, embDim), return_sequences = True))
model.add(Flatten())
model.add(Dense(len(y_train[0]), activation = 'sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])

history = model.fit(x_train, y_train,
epochs = 10, batch_size = 500, validation_data = (x_val,y_val))

#Plotting Results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

