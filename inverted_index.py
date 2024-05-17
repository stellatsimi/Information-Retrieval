import glob
import os
import numpy as np
import nltk 
import math
QUERY = 0

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
tokens = []
inverted_index = {}
vector_space_tfc = []
vector_space_txc = []
doc_length = len(os.listdir("docs"))
vector_space_tfc = [ [] for _ in range(doc_length) ]
vector_space_txc = [ [] for _ in range(doc_length) ]
query_vector = []
listOfDocNames = []
K = 20

def append_queries():
    f = open("Queries_20", "r")
    text = f.read().split('\n')
    return text


def query_weighting(text, i):
    query_vector = []
    query = text[i]
    q = query.split(" ")
    for key in inverted_index.keys():
        if key in q:
            query_vector.append(nfx(q,key))
        else:
            query_vector.append(0)
    return query_vector


def nfx(q, key):
    count = 0
    max = 1
    for term in q:
        if term == key:
            count += 1
        if (count > max):
            max = count
    
    tf = (0.5 * (count / max) + 0.5)
    nt = len(inverted_index[key])
    idf = math.log(doc_length / nt)
    return tf * idf


def append_tokens():
    for filename in glob.glob("docs\*"):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            text = f.read()
            tokens.append(text.lower().split("\n"))
        listOfDocNames.append(os.path.basename(filename))




def cleanup(token_lists):
    cleaned_tokens = [[token for token in token_list if token not in stop_words and len(token) >= 4] for token_list in token_lists]
    return cleaned_tokens
        

def create_inverted_index(inverted_index):
    for i in range(len(listOfDocNames)):
        for index, token in enumerate(tokens[i]):
            if token not in inverted_index:
                inverted_index[token] = {listOfDocNames[i]: [1, [index]]}
            else:
                if i in inverted_index[token]:
                    inverted_index[token][listOfDocNames[i]][0] += 1
                    inverted_index[token][listOfDocNames[i]][1].append(index)
                else:
                    inverted_index[token][listOfDocNames[i]] = [1, [index]]


def numerator_tfc(vector_space):
    for i in range(len(listOfDocNames)):
        for key, value in inverted_index.items():
            if listOfDocNames[i] in value:
                tf = inverted_index[key][listOfDocNames[i]][0] #first element of outer list + 1
                n = len(inverted_index[key]) #length of inner list 
                vector_space[i].append(tfc(tf, n))
            else:
                vector_space[i].append(0) #not zero


def numerator_txc(vector_space):
    for i in range(len(listOfDocNames)):
        for key, value in inverted_index.items():
            if listOfDocNames[i] in value:
                tf = inverted_index[key][listOfDocNames[i]][0] #first element of outer list + 1
                n = len(inverted_index[key]) #length of inner list 
                vector_space[i].append(txc(tf))
            else:
                vector_space[i].append(0) #not zero


def normalize_tfc(vector_space):
    N = doc_length
    for i in range(len(listOfDocNames)):
        product = 0
        for key, value in inverted_index.items():
            if listOfDocNames[i] in value:
                tf = inverted_index[key][listOfDocNames[i]][0]
                #print(tf, n)
            else:
                tf = 0
            n = len(inverted_index[key])
            product += (tf * math.log(N/n) ** 2)
        vector_space[i] = [num / math.sqrt(product) for num in vector_space[i]]


def tfc(tf, n):
    N = doc_length
    return (tf) * math.log(N/n)


def txc(tf):
    return (tf) 


def normalize_txc(vector_space):
    for i in range(len(listOfDocNames)):
        product = 0
        for key, value in inverted_index.items():
            if listOfDocNames[i] in value:
                tf = inverted_index[key][listOfDocNames[i]][0]
                #print(tf, n)
            else:
                tf = 0
            product += (tf ** 2) 
        vector_space[i] = [num / math.sqrt(product) for num in vector_space[i]]


def cosine(vector_space, query):
    similarities = []
    query_norm = np.linalg.norm(query)
    for i in range(len(vector_space)):
        inner_product = np.dot(vector_space[i], query)
        doc_norm = np.linalg.norm(vector_space[i])
        result = inner_product / (query_norm * doc_norm)
        similarities.append(result)
    temp = sorted(similarities)[-K:]
    results = []
    for ele in temp:
        results.append(listOfDocNames[similarities.index(ele)])
    return results




def start_tfc():
    numerator_tfc(vector_space_tfc)
    normalize_tfc(vector_space_tfc)

def start_txc():
    numerator_txc(vector_space_txc)
    normalize_txc(vector_space_txc)

def run_all_queries_tfc():
    queries = append_queries()
    f = open("tfc_results.txt", "w")
    f.truncate()
    for query_num in range(len(queries)):
        query_vector = query_weighting(queries, query_num)
        line = cosine(vector_space_tfc, query_vector)
        for d in line:
            f.write("%s " % d.lstrip("0"))
        f.write("\n")
        
def run_all_queries_txc():
    queries = append_queries()
    f = open("txc_results.txt", "w")
    f.truncate()
    for query_num in range(len(queries)):
        query_vector = query_weighting(queries, query_num)
        line = cosine(vector_space_txc, query_vector)
        for d in line:
            f.write("%s " % d.lstrip("0"))
        f.write("\n")


append_tokens()
tokens = cleanup(tokens)
create_inverted_index(inverted_index)
start_tfc()
run_all_queries_tfc()
start_txc()
run_all_queries_txc()