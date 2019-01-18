import os
import string
import re 
import operator
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from library import clean_text_simple, terms_to_graph, accuracy_metrics

stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

##################################
# read and pre-process abstracts #
##################################

path_to_abstracts = "../data/Hulth2003testing/abstracts/" # fill me!
abstract_names = sorted(os.listdir(path_to_abstracts))

abstracts = []
for counter,filename in enumerate(abstract_names):
    # read file
    with open(path_to_abstracts + '/' + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text = re.sub('\s+', ' ', text)
    abstracts.append(text)
    
    if counter % round(len(abstract_names)/10) == 0:
        print(counter, 'files processed')

abstracts_cleaned = []
for counter,abstract in enumerate(abstracts):
    my_tokens = clean_text_simple(abstract,my_stopwords=stpwds,punct=punct)
    abstracts_cleaned.append(my_tokens)
    
    if counter % round(len(abstracts)/10) == 0:
        print(counter, 'abstracts processed')

###############################################
# read and pre-process gold standard keywords #
###############################################

path_to_keywords = "../data/Hulth2003testing/uncontr/" # fill me!
keyword_names = sorted(os.listdir(path_to_keywords))
   
keywords_gold_standard = []

for counter,filename in enumerate(keyword_names):
    # read file
    with open(path_to_keywords + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    text =  re.sub('\s+', ' ', text) # remove formatting
    text = text.lower()
    # turn string into list of keywords, preserving intra-word dashes 
    # but breaking n-grams into unigrams
    keywords = text.split(';')
    keywords = [keyword.strip().split(' ') for keyword in keywords]
    keywords = [keyword for sublist in keywords for keyword in sublist] # flatten list
    keywords = [keyword for keyword in keywords if keyword not in stpwds] # remove stopwords (rare but may happen due to n-gram breaking)
    keywords_stemmed = [stemmer.stem(keyword) for keyword in keywords]
    keywords_stemmed_unique = list(set(keywords_stemmed)) # remove duplicates (may happen due to n-gram breaking)
    keywords_gold_standard.append(keywords_stemmed_unique)
    
    if counter % round(len(keyword_names)/10) == 0:
        print(counter, 'files processed')

##############################
# precompute graphs-of-words #
##############################
gs = [] ### fill the gap ###
for elem in abstracts:
	my_tokens = clean_text_simple(elem,my_stopwords=stpwds,punct=punct)                   
	gs_pre = terms_to_graph(elem, w=4)
	gs.append(gs_pre)

#gs = [terms_to_graph(elem, w=4) for elem in 

##################################
# keyword extraction with k-core #
##################################

keywords_kc = []  

for counter,g in enumerate(gs):
    core_numbers = dict(zip(g.vs['name'],g.coreness())) # compute core numbers
    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd,c_n in core_numbers.items() if c_n==max_c_n]
    keywords_kc.append(keywords)
    
    if counter % round(len(gs)/10) == 0:
        print(counter)

####################################
# keyword extraction with PageRank #
####################################

my_percentage = 0.33

keywords_pr = []

for counter,g in enumerate(gs):
    pr_scores = zip(g.vs['name'],g.pagerank()) ### fill the gap ### use the .pagerank() method
    pr_scores = sorted(pr_scores, key=operator.itemgetter(1), reverse=True) # rank in decreasing order
    ### fill the gap ### # retain top 'my_percentage' % words as keywords
    numb_to_retain = int(len(pr_scores))*my_percentage
    keywords = [tuple[0] for tuple in pr_scores[:int(numb_to_retain)]]
    print(keywords)
    keywords_pr.append(keywords)  
    
    if counter % round(len(gs)/10) == 0:
        print(counter)

##################################
# keyword extraction with TF-IDF #
##################################

# to ensure same pre-processing as the other methods
abstracts_cleaned_strings = [' '.join(elt) for elt in abstracts_cleaned]

tfidf_vectorizer = TfidfVectorizer(stop_words=stpwds)### fill the gap ### # use TfidfVectorizer passing 'stpwds' as stopwords
doc_term_matrix = tfidf_vectorizer.fit_transform(abstracts) ### fill the gap ###
terms = tfidf_vectorizer.get_feature_names()
vectors_list = doc_term_matrix.todense().tolist()

keywords_tfidf = []

for counter,vector in enumerate(vectors_list):
    terms_weights = zip(terms,vector) # bow feature vector as list of tuples
    ### fill the gap ### # keep only non zero values (the words in the document) and save as object 'nonzero'
    nonzero = [my_tuple for my_tuple in terms_weights if my_tuple[1]!=0] #keep only the non zero values (the words in the doc)

    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True) # rank by decreasing weights
    numb_to_retain = int(len(nonzero)*my_percentage) # retain top 'my_percentage' % words as keywords
    keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]
    
    keywords_tfidf.append(keywords)
    
    if counter % round(len(vectors_list)/10) == 0:
        print(counter)

##########################
# performance comparison #
##########################

perf_kc = []
perf_tfidf = []
perf_pr = []

for idx, truth in enumerate(keywords_gold_standard):
    ### fill the gaps ### # use the 'accuracy_metrics' function
    perf_kc.append(accuracy_metrics(keywords_kc[idx], truth))
    perf_tfidf.append(accuracy_metrics(keywords_tfidf[idx], truth))
    perf_pr.append(accuracy_metrics(keywords_pr[idx], truth))
lkgs = len(keywords_gold_standard)

# print macro-averaged results (averaged at the collection level)
results = {'k-core':perf_kc,'tfidf':perf_tfidf,'PageRank':perf_pr}

for name, result in results.items():
    print(name + ' performance: \n')
    print('precision:', round(100*sum([tuple[0] for tuple in result])/lkgs,2))
    print('recall:', round(100*sum([tuple[1] for tuple in result])/lkgs,2))
    print('F-1 score:', round(100*sum([tuple[2] for tuple in result])/lkgs,2))
    print('\n')
