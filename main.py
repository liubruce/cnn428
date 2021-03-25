import sys
from create_embedding_word2vec import create_word2vec
from create_voc import create_voc
from train_word2vec import word2vec_predict
from os import path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1 == 'create_voc':
            create_voc()
        if arg1 == 'embedding_layer':  # 88%
            if path.exists('vocab.txt') is False:
                create_voc()
            word2vec_predict(True, False)
        if arg1 == 'create_word2vec':
            create_word2vec()
        if arg1 == 'word2vec':  # 57%
            if path.exists('vocab.txt') is False:
                create_voc()
            if path.exists('embedding_word2vec.txt') is False:
                create_word2vec()
            word2vec_predict(False, False)
        if arg1 == 'pre_trained_Glove':  # 77%
            if path.exists('vocab.txt') is False:
                create_voc()
            word2vec_predict(False, True)
        # else:
        #     print(
        #         'Please run the program with either one of three arguments: '
        #         'embedding_layer or word2vec or pre_trained_Glove')
    else:
        print(
            'Please run the program with either one of three arguments: '
            'embedding_layer or word2vec or pre_trained_Glove')
