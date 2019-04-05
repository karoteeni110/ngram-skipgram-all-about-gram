from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

model = Word2Vec.load("w2v.model")
print(model.wv['lēg'])
# [-0.5363539   0.17798892  0.06277177  1.2296044   0.54949623  1.7093421
#  -0.48433432  0.2530322   2.407268   -2.3779027 ]

# load.py
# [ 0.65023184  1.1402951  -0.33691183 -1.8935788  -1.1372805  -0.6000017
#  0.4351502  -0.66633    -1.7164932   0.42037135]
