import re 
import itertools
import operator
import copy
import igraph
import heapq
import nltk
from nltk import pos_tag

def clean_text_simple(text, my_stopwords, punct, remove_stopwords=True, pos_filtering=True, stemming=True):
    text = text.lower()
    text = ''.join(l for l in text if l not in punct) # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +',' ',text) # strip extra white space
    text = text.strip() # strip leading and trailing white space
    tokens = text.split(' ') ### fill the gap ### # tokenize (split based on whitespace)
    if pos_filtering == True:
        # POS-tag and retain only nouns and adjectives
        tagged_tokens = pos_tag(tokens)
        tokens_keep = []
        for item in tagged_tokens:
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        tokens = [item for item in tokens if item not in my_stopwords]### fill the gap ### # filter out stopwords. You may use a list comprehension.
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
    
    return(tokens)


def terms_to_graph(terms, w):
    '''This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurrence within a sliding window of fixed size 'w'.
    '''
    
    from_to = {}
    
    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))
    
    new_edges = []
    
    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
    
    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i] # term to consider
        terms_temp = terms[(i-w+1):(i+1)] # all terms within sliding window
        
        # edges to try
        candidate_edges = []
        for p in range(w-1):
        	candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
			# if not self-edge :
            if try_edge[1] != try_edge[0]:
                if try_edge in from_to:
                    from_to[try_edge] += 1
                ### fill the gap ### # if edge has already been seen, update its weight
                
                ### fill the gap ### # if edge has never been seen, create it and assign it a unit weight     
                else:
                    from_to[try_edge] = 1
    
    # create empty igraph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertex weights
    g.es['weight'] = list(from_to.values()) # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=list(from_to.values())) # weighted degree
    
    return(g)

def unweighted_k_core(g):
    # work on clone of g to preserve g 
    gg = copy.deepcopy(g)    
    
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs['name'],[0]*len(gg.vs)))
    
    i = 0
    
#'''  # while there are vertices remaining in the graph
#    while len(gg.vs)>0:
#        ### fill the gaps ###
#        while 
#        cores_g[gg.vs[index]['name']] = i # assign i as the vertex core number
#        ### fill the gap ### # use the 'delete_vertices' igraph method
#		gg.vs = gg.delete_edges(gg.vs[index]['name'])
#		liste = [gg.get_eid(u,v) for u in gg.vs and for v in gg.vs[index]['name']]
#		gg.es = gg.delete_vertices()'''


    # while there are vertices remaining in the graph
    while len(gg.vs)>0:
        ### fill the gaps ###
		#While there is a vertex with degree less than i
        while [deg for deg in gg.strength() if deg<= i]:
            index = [ind for ind, deg in enumerate(gg.strength()) if deg<=i][0]
            cores_g[gg.vs[index]['name']] = i # assign i as the vertex core number
            gg.delete_vertices(index)

        i += 1
    
    return cores_g

def accuracy_metrics(candidate, truth):
    # note : es and vs methods

    # true positives ('hits') are both in candidate and in truth
    tp = len(set(candidate).intersection(truth))
    
    # false positives ('false alarms') are in candidate but not in truth
    fp = len([element for element in candidate if element not in truth])
    
    # false negatives ('misses') are in truth but not in candidate
    fn = [item for item in truth if item not in candidate] ### fill the gap ###
    
    prec = tp/(tp+fp)
    rec = tp/(tp+len(fn)) ### fill the gap ###
    
    if prec+rec != 0:
        f1 = 2*prec*rec/(prec+rec)
    else:
        f1 = 0
    
    return (prec, rec, f1)
