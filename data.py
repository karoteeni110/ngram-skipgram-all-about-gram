# -*- coding: utf-8 -*-

def readfile(fpath):
    ''' 
    The txt files are processed. '+' are added between morphs.
    '''
    with open(fpath, 'r', encoding = "utf-8") as txtfile:
        print('Reading data...')
        txt = txtfile.readlines()
        wordlist = []        
        for line in txt:
                wordlist.extend(line.strip('\n').strip().split(' ')) 
                        
        # morphs = set(word.split('+') for para in txt for word in para)
        
        # return txt
        return wordlist

if __name__ == "__main__":
    
    readfile('MeBo-123.2015_stamd.txt')

