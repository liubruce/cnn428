import sys
from create_embedding_word2vec import create_word2vec
from create_voc import create_voc
from train_word2vec import word2vec_predict

if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1 == 'create_voc':
            create_voc()
        if arg1 == 'embedding_layer':  # 88%
            word2vec_predict(True, '')
        if arg1 == 'create_word2vec':
            create_word2vec()
        if arg1 == 'word2vec':  # 57%
            word2vec_predict(False, False)
        if arg1 == 'pre_trained_Glove':  # 77%
            word2vec_predict(False, True)
        else:
            print(
                'Please run the program with either one of five arguments: '
                'create_voc or create_word2vec or embedding_layer or word2vec or pre_trained_Glove')
    else:
        print(
            'Please run the program with either one of five arguments: '
            'create_voc or create_word2vec or embedding_layer or word2vec or pre_trained_Glove')
