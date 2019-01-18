# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# here we define our document collection which contains 5 documents
# this is an array of strings
documents = ["Euler is the father of graph theory",
             "Graph theory studies the properties of graphs",
             "Graph theory is cool!",
             "DNA sequences are very complex biological structures",
             "Genes are biological structures that are parts of a DNA sequence",
             "Genes are very important biological structures"]

# create the tf-idf vectors of the document collection
tfidf_vectorizer = TfidfVectorizer()

## DONE: get the matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_matrix = tfidf_matrix.toarray()
A = tfidf_matrix
print(tfidf_matrix)

# apply Singular Value Decomposition
U, S, V = np.linalg.svd(A)
print(U.shape)
print(S.shape)
print(V.shape)

print(S)

print(V.transpose())

# this is the original matrix
print(A)

# keep the first two rows of V
V2 = V[:2,:]
print(V2)

# the matrix after dimensionality reduction
## TODO: get the M matrix
M = np.dot(A, V2.transpose())
print(M)

# plot the results
colors = ['blue','red','black','green','orange','brown']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(6):
    ax.scatter(M[i,0],M[i,1], color=colors[i])
ax.scatter(0,0,color='black')
plt.xlabel('SVD1')
plt.xlabel('Concept 1')
plt.ylabel('Concept 2')
plt.show()
