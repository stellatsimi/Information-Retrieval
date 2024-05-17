import glob
import os
import numpy as np
import nltk 
import math
results = []
relevant_docs = []
K = 20

def open_relevants(file):
    f = open(file, "r")
    text = f.read().split("\n")
    for t in text:
        relevant_docs.append(t)

def initialization(file):
    results = [] 
    f = open(file, "r")
    lines = f.read().split("\n")
    for line in lines:
        results.append(line.split(" "))
    return results

def precisions(QUERY, results):
    count_relevant = 1
    sum_precisions = 0
    for rank, doc in enumerate(results[QUERY], 1):
        if doc in relevant_docs[QUERY]:
            precision = count_relevant / rank 
            sum_precisions += precision
            count_relevant += 1

    average_precision = sum_precisions / count_relevant
    return average_precision


def ap_results_tfc(file, query_file):
    f = open(file, "w")
    q = open(query_file, "r")
    results = initialization("tfc_results.txt")
    for query in range(len(q.read().split("\n"))):
        f.write(str(precisions(query, results)))
        f.write("\n")


def ap_results_txc(file, query_file):
    f = open(file, "w")
    q = open(query_file, "r")
    results = initialization("txc_results.txt")
    for query in range(len(q.read().split("\n"))):
        f.write(str(precisions(query, results)))
        f.write("\n")


def ap_results_colbert(file, query_file):
    f = open(file, "w")
    q = open(query_file, "r")
    results = initialization("colbert_results.txt")
    for query in range(len(q.read().split("\n"))):
        f.write(str(precisions(query, results)))
        f.write("\n")


def calculate_relevance(file_results, file_detailed):
    f = open(file_detailed, "r")
    text = f.read().split("\n")
    for i in range(len(text)):
        text[i] = list(text[i].split(" "))

    g = open(file_results, "r")
    res = g.read().split("\n")
    for i in range(len(res)):
        res[i] = list(res[i].split(" "))

    relevances = []
    for i in range(len(text)):
        relevance = []
        res[i].reverse()
        for z in range(len(res[i])):
            for j in range(0, len(text[i]), 2):
                if res[i][z] == text[i][j]:
                    sum = 0
                    for digit in text[i][j+1]:
                        sum += int(digit)
                    value = math.log(z + 2, 2) # start from position 1 
                    relevance.append(sum / value)
                    break
            else:
                relevance.append(0)
        relevances.append(relevance)
    return relevances


def idcgk(relevances):
    irelevances = []
    for i in relevances:
        irel = sorted(i, reverse=True)
        irelevances.append(irel)
    return irelevances


def dcgk(relevances):
    results = []
    for i in range(len(relevances)):
        result = []
        for j in range(len(relevances[i])):
            value = sum(relevances[i][:j+1])
            result.append(value)
        results.append(result)
    return results


def ndcgk(relevances, file):
    f = open(file, "w")
    dcgk_list = dcgk(relevances)
    irelevances = idcgk(relevances)
    idcgk_list = dcgk(irelevances)
    ndcgk_list = []
    for i in range(len(dcgk_list)):
        res = []
        res = [ j / k for j, k in zip(dcgk_list[i], idcgk_list[i])]
        ndcgk_list.append(res)

    for i in range(len(ndcgk_list)):
        for j in range(len(ndcgk_list[i])):
            f.write(str(ndcgk_list[i][j]))
            f.write("\t")
        f.write("\n")







open_relevants("Relevant_20")
ap_results_tfc("ap_results_tfc.txt", "Queries_20")
ap_results_txc("ap_results_txc.txt", "Queries_20")
ap_results_colbert("ap_results_colbert.txt", "Queries_20")

relevances_tfc = calculate_relevance("tfc_results.txt", "query_doc_relevance.txt")
ndcgk(relevances_tfc, "ndcgk_results_tfc.txt")
relevances_txc = calculate_relevance("txc_results.txt", "query_doc_relevance.txt")
ndcgk(relevances_txc, "ndcgk_results_txc.txt")
relevances_colbert = calculate_relevance("colbert_results.txt", "query_doc_relevance.txt")
ndcgk(relevances_colbert, "ndcgk_results_colbert.txt")
