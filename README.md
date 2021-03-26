# cnn428
1. Download the training and test dataset from : https://raw.githubusercontent.com/jbrownlee/Datasets/master/review_polarity.tar.gz

2. Zip the file and move the directory "txt_sentoken" to the directory where the programs files are exist.

3. Download  pre-trained GloVe vectors from : http://nlp.stanford.edu/data/glove.6B.zip

4. zip the glove.6b.zip to move the text files into the directory glove.6b and then move the directory "glove.6b"
    to the directory where the programs files are exist.

3. Run the program using either of ways:
   
   a. run: "python main.py embedding_layer", which uses the technique - Embedding Layer to learn and evaluate.
   
   b. run: "python main.py word2vec", Word2Vec
   
   c. run: "python main.py pre_trained_Glove", Global Vectors for Word Representation

4. The baseline model is the pre_trained_Glove method and the accuracy is about 76%.