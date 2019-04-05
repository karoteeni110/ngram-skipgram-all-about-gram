from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

model = Word2Vec.load("w2v.model")
print(model.wv['lēg'])
# [-0.19204696  0.42583823 -0.5400337   1.4169201  -0.6937777  -1.3581121
#  -0.91482985 -0.80812997 -1.0136793  -0.65194935]

# V.S. results got from PyTorch way:
# [ 0.65023184  1.1402951  -0.33691183 -1.8935788  -1.1372805  -0.6000017
#  0.4351502  -0.66633    -1.7164932   0.42037135]
