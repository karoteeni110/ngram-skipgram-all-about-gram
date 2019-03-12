# -*- coding: utf-8 -*-
"""
    LEAVE MESSAGE HERE
"""
from nltk.tokenize import RegexpTokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#-- hyperparameters --#
WINDOW_SIZE = 2
EMBEDDING_DIM = 100
N_NEGS = 10
N_EPOCHS = 30
LEARNING_RATE = 0.01

toy_corpus = [
    'You may work either independently or in a group',
    'We have five suggestions for topics and practical projects',
    'We will provide datasets for each practical project',
    'You can also choose your own topic and suggest a project or choose and existing topic and suggest your own project based on the topic'
]


#-- model --#
class SGNS(nn.Module): #Skipgram (without negative sampling for now)
    def __int__(self, embedding_dim, vocab_size):
        super(SGNS, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embed_vec = self.embed(x)
        log_probs = F.log_softmax(self.linear(embed_vec))
        return log_probs


#-- auxilary functions --#
def get_context_words(window_size, input_word):
    context_word_idx = None
    return context_word_idx

def word2idx(word):
    w2d = {w: idx for (idx, w) in enumerate(vocab)}
    idx = indices[word]
    return idx

def get_onehot(word):
    """
    # A list of tokenized corpus
    corpus_tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in corpus:
        sentence_tokens = tokenizer.tokenize(sentence)
        corpus_tokens.append(sentence_tokens)
    
    """
    onehot = torch.zeros(len(vocab),dtype=torch.long)
    onehot[indices[word]] = 1
    return onehot

def seq_and_vocab(corpus): 
    seqs=[]
    for sentence in corpus:
        seqs += sentence.split(' ') 
    vocab = set(seqs)
    return seqs,vocab

if __name__=='__main__':

    # Get one-hot
    seqs, vocab = seq_and_vocab(toy_corpus)
    

    #-- initialization --#
    model = SGNS(EMBEDDING_DIM, len(vocab))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)

    #-- training --#
    for epoch in range(N_EPOCHS):
        total_loss = 0

        for sentence in corpus:
            context_lst = []
            center_w = None
            for context_w in context_lst: 
                log_probs = model(get_onehot(center_w))
                loss = loss_function(log_probs, word2idx(context_w))
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    #-- testing --#
    with torch.no_grad():
        for sentence in corpus: # maybe use a function 
            word_vec = None