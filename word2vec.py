# -*- coding: utf-8 -*-
"""
    LEAVE MESSAGE HERE
"""
from random import choice, random, shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from memory_profiler import profile
import pickle

# from collections import Counter

#-- hyperparameters --#
WINDOW_SIZE = 2
EMBEDDING_DIM = 10
# N_NEGS = 10
N_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 3000

toy_corpus = [
    'You may work either independently or in a group',
    'We have five suggestions for topics and practical projects',
    'We will provide datasets for each practical project',
    'You can also choose your own topic and suggest a project or choose and existing topic and suggest your own project based on the topic'
]
#CORP = toy_corpus
CORP = open('UD-FI-untagged.txt', 'r').readlines()
# print(len(CORP))
# 18619

#-- model --#
class SGnoNS(nn.Module): #Skipgram (without negative sampling for now)
    def __init__(self, embedding_dim, vocab_size):
        super(SGnoNS, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size 
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size) # Is this size correct?

    def forward(self, x):
        out = self.embed(x)
        embed_vec = out.view(BATCH_SIZE,-1) # [vocab_size, embedding_dim].view(1,-1); NOT [1, embedding_dim]!
        log_probs = F.log_softmax(self.linear(embed_vec), dim=1)
        return log_probs

#-- auxilary functions --#
def seq_and_vocab(corpus): 
    '''
    sent_tokens: A list. Containing nested lists, where contains tokens in each sentence.
    '''
    sent_tokens=[]
    corpus_tokens=[]
    for sentence in corpus:
        lst = sentence.split()
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

def get_word_pairs(sent_tok, window_size): 
    '''
    window_size: | pos(farthest context word) - pos(center) | 
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

if __name__=='__main__':
    # Exclude sparse words
    # TBD

    # Get vocab, idx, word, trainset for next steps
    sent_tokens, vocab = seq_and_vocab(CORP)
    w2d = {w: idx for (idx, w) in enumerate(vocab)}
    i2w = {idx:w for (idx, w) in enumerate(vocab)}
    print('Reading...')
    
    trainset=get_word_pairs(sent_tokens, WINDOW_SIZE)

    # Build Unigram Distribution**0.75
    # TBD
    
    # Negative Sampling
    # TBD
    
    #-- initialization --#
    print('Initialzing..')
    model = SGnoNS(EMBEDDING_DIM, len(vocab))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)

    #-- training --#
    print('Training..')
    for epoch in range(N_EPOCHS):
        total_loss = 0  
        shuffle(trainset)

        # print(len(trainset))
        # 974794
        #for center,context in trainset:
        for i in range(0,int(len(trainset)/BATCH_SIZE)):
            minibatch = trainset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            mb_center = torch.cat([word2idx(center)
                        for center, context in minibatch], dim=0)
            # log_probs = model(word2idx(center).view(1,-1))   
            log_probs = model(mb_center)
            mb_context = torch.cat([word2idx(context)
                        for center, context in minibatch], dim=0)
            
            loss = loss_function(log_probs, mb_context)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #-- Report loss & save embed after every epoch --#
        with torch.no_grad():
            print('Epoch', epoch,'Loss:', total_loss)

            matrix = {}
            ebd_matrix = model.embed.weight.data.numpy()
            for word in vocab:
                matrix[word] = ebd_matrix[word2idx(word)]
            pickle.dump(matrix,open('UD-FI-Pytorch-emb.EP%d.pkl' % epoch,'wb'))
        
    # Sanity check:
    # print(model.embed.weight.data.numpy()[word2idx('un')])

    