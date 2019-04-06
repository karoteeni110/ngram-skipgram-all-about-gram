# -*- coding: utf-8 -*-
'''
A parallel implementation of word2vec in gensim.
Some snippets adapted from the tutorial: 
https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb 
'''
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from random import shuffle

WINDOW_SIZE = 2
EMBEDDING_DIM = 10
EPOCHS = 10
WORKERS = 1 # cpu_count()
# MIN_COUNT = 1
# SAMPLE_RATE = 6e-1
ALPHA = 0.01
# MINALPHA = 0.0007
# NEG_NUM = 20
SEED_NUM = 1

def read_input(infile):
    '''
    This function builds a generator.
    list(read_input(infile)) is a list of sentence words:

    [['hē', 'hârr', 'blōts', 'ėn', 'büx', 'antrock', 
    'un', 'wēr', 'sō', 'an', 'finster', 'ranpedd', 'ėm', 
    'hârr', 'ėn', 'dummerhaftig', 'drōm', 'bangmook'],
    ['schiet', 'ōk']]

    '''
    with open(infile,'rb') as f:
        for line in f:
            yield simple_preprocess(line)

CORP = list(read_input('newtxt.txt')) 
shuffle(CORP)
# print(CORP[:2])
# exit(0)

#-- Model setup --#
w2v_model = Word2Vec(# min_count=MIN_COUNT,
                     window=WINDOW_SIZE,
                     size=EMBEDDING_DIM,
                     seed=SEED_NUM,
                     # sample=SAMPLE_RATE, 
                     alpha=ALPHA, 
                     # min_alpha= MINALPHA, 
                     # negative=NEG_NUM,
                     workers=WORKERS)
w2v_model.build_vocab(CORP, progress_per=10000) #Building a vocab table
# Check what's in vocab:
# word_vectors = w2v_model.wv.vocab
# print(list(word_vectors.keys())[:3])

#-- Training --#
w2v_model.train(CORP, 
                total_examples=w2v_model.corpus_count, 
                epochs=EPOCHS)

w2v_model.save('w2v.model')

# Check similar words
# w1= ['hârr']
# w2v_model.wv.most_similar(positive=w1)