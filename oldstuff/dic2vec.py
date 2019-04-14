d, w2d = pickle.load(open('myemb.EP9.pkl','rb'))
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils
import gensim

m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=10)
m.vocab = d
m.vectors = np.array(list(d.values()))
def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

my_save_word2vec_format(binary=True, fname='train.bin', total_vec=len(d), vocab=m.vocab, vectors=m.vectors)