# -*- coding: utf-8 -*-
"""
    LEAVE MESSAGE HERE
"""
import os
from random import choice, random, shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

from data import readfile

#-- hyperparameters --#
WINDOW_SIZE = 3
EMBEDDING_DIM = 10
# N_NEGS = 10
N_EPOCHS = 1
REPORT_EVERY = 1
LEARNING_RATE = 0.01

'''
toy_corpus = [
    'You may work either independently or in a group',
    'We have five suggestions for topics and practical projects',
    'We will provide datasets for each practical project',
    'You can also choose your own topic and suggest a project or choose and existing topic and suggest your own project based on the topic'
]

CORP = toy_corpus
'''
CORP = readfile('MeBo-123.2015_stamd.txt')

#-- model --#
class SGNS(nn.Module): #Skipgram (without negative sampling for now)
    def __init__(self, embedding_dim, vocab_size):
        super(SGNS, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size 
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(vocab_size * embedding_dim, vocab_size) # Is this input size correct?

    def forward(self, x):
        out = self.embed(x)
        embed_vec = out.view(1,-1) # [vocab_size, embedding_dim].view(1,-1); NOT [1, embedding_dim]!
        log_probs = F.log_softmax(self.linear(embed_vec), dim=1)
        return log_probs
    

#-- auxilary functions --#
def seq_and_vocab(corpus): 
    '''
    corpus: A list. Containing sentences as elements. Example: toy_corpus
    sent_tokens: A list. Containing nested lists, where contains tokens in each sentence
    
    '''
    sent_tokens=[]
    corpus_tokens=[]
    for sentence in corpus:
        # lst = sentence.split() # Split by space delimiter
        # lst = list(sentence)  # Split to chars
        lst = [sentence[i:i+5] for i in range(0, len(sentence), 5)] # n-gram
        sent_tokens.append(lst)
        corpus_tokens += lst 
    vocab = set(corpus_tokens)
    return sent_tokens, vocab

def word2idx(word):
    idx = w2d[word]
    return torch.tensor([idx], dtype = torch.long)

def idx2word(idx):
    word = i2w[idx]
    return word

def get_onehot(word):
    """
    # Tokenization
    corpus_tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in corpus:
        sentence_tokens = tokenizer.tokenize(sentence)
        corpus_tokens.append(sentence_tokens)
    
    """
    onehot = torch.zeros(len(vocab), dtype=torch.long)
    onehot[word2idx(word)] = 1
    return onehot

def get_word_pairs(sent_tok, window_size): 
    ''' 
    window_size: | pos(farthest context word) - pos(center) | 
    sent_tok: ['You', 'are', 'welcome']
    window_size: int
    '''
    word_pairs = []
    for sentence in sent_tok:
        for center_pos in range(len(sentence)):
            center = [sentence[center_pos]] * (window_size *2)
            context = []    
            for context_pos in range(center_pos - window_size, center_pos + window_size+1):        
                if context_pos>=0 and context_pos <= len(sentence)-1 and context_pos != center_pos:
                    context.append(sentence[context_pos])        
                else: 
                    continue
            word_pairs += [[c,t] for (c,t) in zip(center, context)]
        '''            
            print(center[0], context)
        print(sentence)                
        print(word_pairs)
        exit(0)
        '''        
    return word_pairs    

def embed_vec(word):
    vec = embed_matrix[word2idx(word)]
    return vec


if __name__=='__main__':
    # Exclude sparse words
    # TBD

    # Get vocab, idx, word, trainset for next steps
    sent_tokens, vocab = seq_and_vocab(CORP)
    w2d = {w: idx for (idx, w) in enumerate(vocab)}
    i2w = {idx:w for (idx, w) in enumerate(vocab)}
    trainset = get_word_pairs(sent_tokens, WINDOW_SIZE)
    print('Data loaded.')

    # -- Check point -- #
    # print(sent_tokens[:4])
    # for i in range(7):
    #     print(vocab.pop()) 
    # print(trainset[:4])
    
    # Build Unigram Distribution**0.75
    # TBD
    
    # Negative Sampling
    # TBD
    
    # -- Check point -- #
    # print(get_onehot('e').shape)
    # exit(0)
    
    #-- initialization --#
    model = SGNS(EMBEDDING_DIM, len(vocab))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
    print('Initialization compeleted.')

    #-- training --#
    print('Training...')
    for epoch in range(N_EPOCHS):
        total_loss = 0  
        shuffle(trainset)
        
        # Check point
        # print(len(trainset))
        # exit(0)
        
        for center,context in trainset:
            log_probs = model(get_onehot(center))
            
            # print(center)
            # print('input', get_onehot(center))
            # print('logprob',log_probs, log_probs.shape) 
            # exit(0)
            
            loss = loss_function(log_probs, word2idx(context))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            
        # -- Training report -- #
        if ((epoch+1) % REPORT_EVERY) == 0:
            print('epoch: %d, loss: %.4f ' % (epoch, total_loss*100/len(trainset)))  
            #, train acc: %.2f' % #, 
                                #100.0 * correct / len(trainset)))


    #-- After training --#
    with torch.no_grad():
        embed_matrix = model.embed.weight.data.numpy()
        print('Loss:', total_loss)
        testlist = ['š','l', 'P', 'u']
        for testword in testlist:
            print(embed_matrix[word2idx(testword)])
