from string import punctuation
from os import listdir
from gensim.models import Word2Vec
from doc_base import load_doc

# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
    clean_lines = list()
    lines = doc.splitlines()
    for line in lines:
        # split into tokens by white space
        tokens = line.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        clean_lines.append(tokens)
    return clean_lines


# load all docs in a directory
def embedding_process_docs(directory, vocab, is_trian):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)
        # add lines to list
        lines += doc_lines
    return lines


def create_word2vec():
    vocab_filename = 'vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    # load training data
    positive_docs = embedding_process_docs('txt_sentoken/pos', vocab, True)
    negative_docs = embedding_process_docs('txt_sentoken/neg', vocab, True)
    sentences = negative_docs + positive_docs
    print('Total training sentences: %d' % len(sentences))

    # train word2vec model
    model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
    # summarize vocabulary size in model
    words = list(model.wv.vocab)
    print('Vocabulary size: %d' % len(words))

    # save model in ASCII (word2vec) format
    filename = 'embedding_word2vec.txt'
    model.wv.save_word2vec_format(filename, binary=False)