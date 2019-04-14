import pickle
from scipy.spatial.distance import cosine

matrix1, w2d = pickle.load(open('myemb.EP5.pkl','rb'))
matrix2, w2d = pickle.load(open('myemb.EP9.pkl','rb'))
matrix3 = pickle.load(open('UD-FI-Pytorch-emb.EP0.pkl','rb'))
# print(matrix1['lēg'])
# print(matrix2['lēg'])
a, b, c ='hyvä', 'kiva', 'huono'
print('A:',a, 'B:',b, 'C:',c)
print('Cosine distance of A and B:', cosine(matrix3[a],matrix3[b]))
print('Cosine distance of A and C:', cosine(matrix3[a],matrix3[c]))

# [ 0.6498083   1.1406335  -0.3368125  -1.8939863  -1.137307   -0.5996181
#  0.43552572 -0.66676354 -1.717217    0.4205264 ]
# [ 0.65023184  1.1402951  -0.33691183 -1.8935788  -1.1372805  -0.6000017
#  0.4351502  -0.66633    -1.7164932   0.42037135]
