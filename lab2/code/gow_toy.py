import string
# nltk > 3.2.1 is required
# execute nltk.download('stopwords'), nltk.download('maxent_treebank_pos_tagger'), nltk.download('averaged_perceptron_tagger') if you haven't already
from nltk.corpus import stopwords

#import os
#os.chdir() # change working directory to where functions are

from library import clean_text_simple, terms_to_graph, unweighted_k_core

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = '''A method for solution of systems of linear algebraic equations 
with m-dimensional lambda matrices. A system of linear algebraic 
equations with m-dimensional lambda matrices is considered. 
The proposed method of searching for the solution of this system 
lies in reducing it to a numerical system of a special kind.'''

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)
                              
g = terms_to_graph(my_tokens, w=4)
    
# number of edges
print(len(g.es))

# the number of nodes should be equal to the number of unique terms
assert len(g.vs) == len(set(my_tokens))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)

for w in range(2,10):    
    ### fill the gap ### # build a graph-of-words g
    g = terms_to_graph(my_tokens, )
    print(g.density())
    
# decompose g
core_numbers = unweighted_k_core(g)
print(core_numbers)

# compare with igraph method
print(dict(zip(g.vs['name'],g.coreness())))

# retain main core as keywords
max_c_n = max(list(core_numbers.values()))
print(max_c_n)
print(list(core_numbers.values()))
keywords = [list(core_numbers.items())[count][0] for count, item in enumerate(list(core_numbers.values())) if item == max_c_n] ### fill the gap ### # you may use a list comprehension
print(keywords)
