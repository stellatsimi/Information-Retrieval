import matplotlib.pyplot as plt
import numpy as np

def plot_ap(tfc_file, txc_file, colbert_file):
    f = open(tfc_file, "r")
    result_tfc = f.read().split("\n")
    result_tfc = [float(x) for x in result_tfc]

    g = open(txc_file, "r")
    result_txc = g.read().split("\n")
    result_txc = [float(x) for x in result_txc]

    h = open(colbert_file, "r")
    result_colbert = h.read().split("\n")
    result_colbert = [float(x) for x in result_colbert]
    

    plt.plot(result_tfc, color="#11009E", marker="o", mfc="#EEF5FF",label="tfc")
    plt.plot(result_txc, color="#86B6F6", marker="o", mfc="#EEF5FF",label="txc")
    plt.plot(result_colbert, color="#176B87", marker="o", mfc="#EEF5FF",label="colbert")
    plt.xticks(range(1, 20, 1))
    plt.ylabel("metric")
    plt.xlabel("query")
    plt.title("Average Precision")
    plt.legend()
    plt.show()

def plot_ndcgk(tfc_file, txc_file, colbert_file):
    f = open(tfc_file, "r")
    results_tfc = []
    lines = f.read().split("\n")
    for line in lines:
        results_tfc.append(line.split("\t"))
    for i in results_tfc:
        i.pop() # teleytaios xarakthras kenos logw tab
    for i in range(len(results_tfc)):
        results_tfc[i] = [float(x) for x in results_tfc[i]]

    g= open(txc_file, "r")
    results_txc = []
    lines = g.read().split("\n")
    for line in lines:
        results_txc.append(line.split("\t"))
    for i in results_txc:
        i.pop() # teleytaios xarakthras kenos logw tab
    for i in range(len(results_txc)):
        results_txc[i] = [float(x) for x in results_txc[i]]

    l = open(colbert_file, "r")
    results_colbert = []
    lines = l.read().split("\n")
    for line in lines:
        results_colbert.append(line.split("\t"))
    for i in results_colbert:
        i.pop() # teleytaios xarakthras kenos logw tab
    for i in range(len(results_colbert)):
        results_colbert[i] = [float(x) for x in results_colbert[i]]
    indexes = list(range(20))

    for i in range(20):
        plt.plot(indexes, results_tfc[i],color="#11009E", marker="o",mfc="#EEF5FF", label="tfc")
        plt.plot(indexes,results_txc[i], color="#86B6F6", marker="o",mfc="#EEF5FF", label="txc")
        plt.plot(indexes,results_colbert[i], color="#176B87", marker="o", mfc="#EEF5FF", label="colbert")
        plt.xticks(range(1, 21, 1))
       # plt.yticks(range(0, 2, ))
        plt.yticks(np.arange( 0,1.2,step=0.2))
        plt.ylabel("metric")
        plt.xlabel("query results")
        plt.title("NDCGK for query")
        plt.legend()
        plt.show()




plot_ap("ap_results_tfc.txt", "ap_results_txc.txt", "ap_results_colbert.txt")
plot_ndcgk("ndcgk_results_tfc.txt", "ndcgk_results_txc.txt", "ndcgk_results_colbert.txt")
