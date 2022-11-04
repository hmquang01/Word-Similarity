import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from problem.word_similarity import WordSimilarity
import numpy as np

def evaluate_similarity():
    global vector_arr, word_arr
    visim400 = pd.read_csv("datasets/ViSim-400/ViSim-400.txt", delimiter="\t")
    cosine_builtin_sim = []
    pearson_sim = []
    spearman_sim = []
    cosine_sim = []
    for _,row in visim400.iterrows():
        word1 = row["Word1"]
        word2 = row["Word2"]
        if word1 in word_arr and word2 in word_arr:
            word1_vec = vector_arr[word_arr.index(word1)]
            word2_vec = vector_arr[word_arr.index(word2)]
            cosine_builtin_sim.append(round(WordSimilarity.cosine_builtin(word1_vec, word2_vec), 3))
            pearson_sim.append(round(WordSimilarity.pearson(word1_vec, word2_vec), 3))
            spearman_sim.append(round(WordSimilarity.spearman_rank(word1_vec, word2_vec), 3))
            cosine_sim.append(round(WordSimilarity.cosine(word1_vec, word2_vec), 3))
        else:
            cosine_builtin_sim.append("None")
            pearson_sim.append("None")
            spearman_sim.append("None")
            cosine_sim.append("None")

    visim400["Cosine Builtin"] = cosine_builtin_sim
    visim400["Pearson"] = pearson_sim
    visim400["Spearman"] = spearman_sim
    visim400["Cosine"] = cosine_sim
    visim400.to_csv("visim-400.csv", index=False)

def k_nearest_words(query, k, sim_function):
    candidates = []
    for word in word_arr:
        candidates.append((sim_function(vector_arr[word_arr.index(word)], vector_arr[word_arr.index(query)]), word))
    candidates.sort(reverse=True)
    candidates = [candidate[1] for candidate in candidates[:k]]
    candidates = "\n".join(candidates)
    return candidates

if __name__ == '__main__':
    f = open("./word2vec/W2V_150.txt", "r", encoding="UTF-8")
    arr = []
    for i in f:
        arr.append(i.rstrip())

    word_arr = []
    vector_arr = []

    for i in range(len(arr)):
        word = arr[i].split("  ")[0]
        vector = arr[i].split("  ")[1]
        vector = vector.split(" ")
        vector = [float(x) for x in vector]

        word_arr.append(word)
        vector_arr.append(vector)

    vector_arr = np.array(vector_arr)
    evaluate_similarity()
    word = "ngăn_nắp"
    print("Cosine")
    print(k_nearest_words(word, 10, WordSimilarity.cosine))
    print("\n")
    print("Pearson")
    print(k_nearest_words(word, 10, WordSimilarity.pearson))
    print("\n")
    print("Spearman_rank")
    print(k_nearest_words(word, 10, WordSimilarity.spearman_rank))
    print("\n")
