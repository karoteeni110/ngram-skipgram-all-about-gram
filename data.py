
import nltk 

def readfile(fname):
    with open(fname, 'r') as txtfile:
        txt = txtfile.read()
        return txt.split()

