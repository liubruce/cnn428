from numpy import array
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from doc_base import load_doc, process_docs, get_weight_matrix, glove_load_embedding
from os import path

# load embedding as a dict
def word2vec_load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding


def word2vec_predict(if_layer, if_glove):
    vocab_filename = 'vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    # load all training reviews
    if path.exists('txt_sentoken/neg') is False:
        print('Please make sure the files of training set and test set are exist!')
        exit(-1)
    positive_docs = process_docs('txt_sentoken/pos', vocab, True)
    negative_docs = process_docs('txt_sentoken/neg', vocab, True)
    train_docs = negative_docs + positive_docs

    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(train_docs)
    # pad sequences
    max_length = max([len(s.split()) for s in train_docs])
    Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define training labels
    ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

    # load all test reviews
    positive_docs = process_docs('txt_sentoken/pos', vocab, False)
    negative_docs = process_docs('txt_sentoken/neg', vocab, False)
    test_docs = negative_docs + positive_docs
    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(test_docs)
    # pad sequences
    Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define test labels
    ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1

    embedding_layer = Embedding(vocab_size, 100, input_length=max_length)
    if if_layer is False:
        # load embedding from file
        if if_glove is True:
            if path.exists('glove.6B/glove.6B.100d.txt') is False:
                print('Please make sure the files of pre-trained GloVe vectors are exist!')
                exit(-1)
            raw_embedding = glove_load_embedding('glove.6B/glove.6B.100d.txt')
        else:
            raw_embedding = word2vec_load_embedding('embedding_word2vec.txt')
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
        # create the embedding layer
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

    # define model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(Xtrain, ytrain, epochs=10, verbose=2)
    # evaluate
    loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    print('Test Accuracy: %f' % (acc * 100))

