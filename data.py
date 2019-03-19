# -*- coding: utf-8 -*-

def onesentperline(fpath):
    ''' 
    The txt files are processed. '+' are added between morphs.
    '''
    newfile = open('newtxt.txt','w')

    with open(fpath, 'r', encoding = "utf-8") as txtfile:
        print('Reading data...')
        txt = txtfile.readlines()
        wordlist = []        
        for line in txt:
                wordlist.append(line.strip('\n').strip().split(' ')) 
        
        for word in wordlist:
                if word in ['“', '„', '’', ',']:
                        continue
                if word not in ['.', '!', '?']:
                        newfile.write(word + ' ')
                else:
                        newfile.write('\n') 
        
def seq_and_vocab(corpus, n): 
    '''
    corpus: A list of words. ['\ufeffpeter', 'neuber', ',', 'meldörp-bȫker', '1', '', 'verschēden', 'schrieverslüüd', '']
    sent_tokens: A list 
    
    '''
    sent_tokens=[]
    corpus_tokens=[]
    for word in corpus:
        # lst = word.split() # Split by space delimiter
        # lst = list(word)  # Split to chars
        word = '<%s>' % word
        ngram = [ word[i:i+n] for i in range( len(word)-n+1 ) if i+n <= len(word) ] # n-gram
        sent_tokens.append(ngram) # {word: ngram}
        corpus_tokens += ngram

    vocab = set(corpus_tokens)
    return sent_tokens, vocab
    # print(sent_tokens[:5], len(vocab)) # 5gram
    # [['\ufeffpete', 'r'], ['neube', 'r'], [','], ['meldö', 'rp-bȫ', 'ker'], ['1']] 12544

if __name__ == "__main__":
    
    onesentperline('newtxt.txt')

