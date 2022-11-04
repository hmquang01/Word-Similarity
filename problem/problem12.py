import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from problem.word_similarity import WordSimilarity

def evaluate_similarity():
    global word2vec, vocab
    visim400 = pd.read_csv("datasets/ViSim-400/ViSim-400.txt", delimiter="\t")
    cosine_builtin_sim = []
    pearson_sim = []
    spearman_sim = []
    cosine_sim = []
    for _,row in visim400.iterrows():
        word1 = row["Word1"]
        word2 = row["Word2"]
        if word1 in vocab and word2 in vocab:
            cosine_builtin_sim.append(round(WordSimilarity.cosine_builtin(word2vec[word1], word2vec[word2]), 3))
            pearson_sim.append(round(WordSimilarity.pearson(word2vec[word1], word2vec[word2]), 3))
            spearman_sim.append(round(WordSimilarity.spearman_rank(word2vec[word1], word2vec[word2]), 3))
            cosine_sim.append(round(WordSimilarity.cosine(word2vec[word1], word2vec[word2]), 3))
        else:
            cosine_builtin_sim.append("None")
            pearson_sim.append("None")
            spearman_sim.append("None")
            cosine_sim.append("None")

    visim400["Cosine Builtin"] = cosine_builtin_sim
    visim400["Pearson"] = pearson_sim
    visim400["Spearman"] = spearman_sim
    visim400["Cosine"] = cosine_sim
    visim400.to_csv("problem/visim-400.csv", index=False)

def k_nearest_words(query, k, sim_function):
    global word2vec, vocab
    candidates = []
    for word in vocab:
        candidates.append((sim_function(word2vec[word], word2vec[query]), word))
    candidates.sort(reverse=True)
    candidates = [candidate[1] for candidate in candidates[:k]]
    candidates = "\n".join(candidates)
    return candidates

def show_mean_plot():
    global word2vec, vocab
    mean_embedding = []
    for word in vocab:
        mean_embedding.append(word2vec[word].sum() / 150)
    sns.displot(mean_embedding)
    plt.show()

if __name__ == '__main__':
    word2vec = KeyedVectors.load_word2vec_format('./word2vec/W2V_150.txt')
    vocab = word2vec.index_to_key
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
