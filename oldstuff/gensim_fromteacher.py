from gensim.models import Word2Vec
from random import randint, shuffle

EPOCHS=100

data = open("newtxt.txt").read().split('\n')
data = [line.strip(' ').split(' ') for line in data]

for i in range(5):
    shuffle(data)
    model = Word2Vec(data, size=100, window=5, min_count=1, workers=4, seed=randint(1,100))
    model.train(data, total_examples=len(data), epochs=EPOCHS)

    print("Most similar wordforms to %s in iteration %u:" % ("lēg",i))
    for wf, score in model.most_similar("lēg"):
        print(wf,score)
    print()
