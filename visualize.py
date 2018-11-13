from gensim.models.fasttext import FastText
from gensim.scripts.word2vec2tensor import word2vec2tensor

model = FastText.load('model.bin')

# Taken from: https://www.kdnuggets.com/2018/04/robust-word2vec-models-gensim.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

keywords = ['orange', 'heart', 'ebola', 'flu', 'smoke', 'diabetes']
similar_words = {search_term: [item[0] for item in model.wv.most_similar([search_term], topn = 5)]
                  for search_term in keywords}
print(similar_words)

words = sum([[k] + v for k, v in similar_words.items()], [])
wvs = model.wv[words]

tsne = TSNE(n_components = 2, random_state = 0, n_iter = 10000, perplexity = 2)
np.set_printoptions(suppress = True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize = (14, 8))
plt.scatter(T[:, 0], T[:, 1], c = 'blue', edgecolors = 'r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
  plt.annotate(label, xy = (x + 1, y + 1), xytext = (0, 0), textcoords = 'offset points')
plt.show()