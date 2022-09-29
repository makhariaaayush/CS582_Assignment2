import os
import re
import string
import math
from nltk.tokenize import word_tokenize
import collections
from nltk.corpus import stopwords
from nltk import PorterStemmer
from operator import itemgetter

folder = '/Users/aayushmakharia/Desktop/cranfieldDocs'
query_files = '/Users/aayushmakharia/Desktop/queries.txt'

def process_document(folder):
    files = os.listdir( folder )
    corpus = {}
    for i,file in enumerate(files):
            data = open(os.path.join(folder, file), 'r').read()
            data = preprocessing( data )
            corpus[i+1] = data

    for key, value in corpus.items():
        counter = collections.Counter( value )
        corpus[key] = dict( counter )

    return corpus
def preprocessing(corpus):
    corpus = re.sub( re.compile( '[0-9]' ), '', corpus )
    corpus = re.sub( re.compile( '<.*>' ), '', corpus )
    tokens = word_tokenize( corpus )
    token_remove_punctuation = []
    for word in tokens:
        words =word.translate( str.maketrans( '', '', string.punctuation ) )
        token_remove_punctuation.append(words)
    token_remove_stopwords = []
    stops = set( stopwords.words( 'english' ) )
    for word in token_remove_punctuation:
        if not word in stops:
            token_remove_stopwords.append(word)
    token_lowercase = []
    for word in token_remove_stopwords:
        if word != '':
            token_lowercase.append(word.lower())
    ps = PorterStemmer()
    token_stemmer = []
    for word in token_lowercase:
        token_stemmer.append(ps.stem(word))
    token_length = []
    for word in token_stemmer:
        if len(word) > 2:
            token_length.append(word)
    token_final = token_length
    return token_final
documents = process_document(folder)


tf = {}
for key, val in documents.items():
    for word in val:
        if word not in tf:
            tf[word] = {}
        tf[word][key] = documents[key][word]

idf = {}
N = len( documents )

for key, val in documents.items():
    for word in val:
        if word not in idf:
            count = len( tf[word] )
            idf[word] = math.log2( N / count )

tf_idf = {}
for word in tf:
    for i in tf[word]:
        if i in tf_idf:
            tf_idf[i] += (tf[word].get( i, 0 ) * idf.get( word, 0 )) ** 2
        else:
            tf_idf[i] = (tf[word].get( i, 0 ) * idf.get( word, 0 )) ** 2

for key in tf_idf:
    tf_idf[key] = math.sqrt( tf_idf[key] )




# query_file = os.listdir( query_files )
query = {}
data_queries = open( os.path.join(query_files), 'r' ).read().split('\n')
for i,q in enumerate(data_queries):
    q = preprocessing(q)
    query[i + 1] = q




for key, value in query.items():
    counter = collections.Counter( value )
    query[key] = dict( counter )

tf_query = {}
for key, val in query.items():
    for word in val:
        if word not in tf_query:
            tf_query[word] = {}
        tf_query[word][key] = query[key][word]



tf_idf_query = {}
for word in tf_query:
    for i in tf_query[word]:
        if i in tf_idf_query:
            tf_idf_query[i] += (tf_query[word].get( i, 0 ) * idf.get( word, 0 )) ** 2
        else:
            tf_idf_query[i] = (tf_query[word].get( i, 0 ) * idf.get( word, 0 )) ** 2

for key in tf_idf_query:
    tf_idf_query[key] = math.sqrt( tf_idf_query[key] )



cosine_similarity = {}
for q in query:
    cosine_similarity[q] = {}
    for terms_query, tf_value_query in query[q].items():
        if terms_query in tf:
            for d, d_tf in tf[terms_query].items():
                numer = (d_tf * idf[terms_query]) * (tf_value_query * idf[terms_query])
                deno = tf_idf[d] * tf_idf_query[q]
                if d in cosine_similarity[q]:
                    cosine_similarity[q][d] += (numer / deno)
                else:
                    cosine_similarity[q][d] = (numer / deno)

    sorted_desc =  sorted( cosine_similarity[q].items(), key=itemgetter( 1 ), reverse=True )
    # print(sorted_desc)

    sorted_desc = dict(sorted_desc)
    cosine_similarity[q] = sorted_desc


# Relevant file
relevant = []
relevant_path = '/Users/aayushmakharia/Desktop/relevance.txt'
relevance_docs = open(relevant_path, 'r')
lines = relevance_docs.read().split('\n')

for line in lines:
    relevant.append(line)


relevance = {}

for val in relevant:
    id_query, doc_id = val.split(' ')
    id_query = int(id_query)
    try:
        relevance[id_query].append(int(doc_id))
    except:
        relevance[id_query] = [int(doc_id)]

document_retrieved = {id_query: list(docs.keys()) for id_query, docs in cosine_similarity.items()}



def computep(relevant , retrieved):

    tp = len([doc_id for doc_id in relevant if doc_id in retrieved])
    fp = len(retrieved) - tp
    fn = len(relevant) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (precision,recall)



ir = {}
for retrieved_N_values in [10, 50, 100, 500]:
    precision = 0
    recall = 0
    for qu in query:

        p,r = computep(relevance[qu],document_retrieved[qu][:retrieved_N_values])

        precision += p
        recall += r

        ir[retrieved_N_values] = (precision/10,recall/10)

print(ir)