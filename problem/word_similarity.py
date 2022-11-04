from scipy import stats, spatial
import numpy as np

class WordSimilarity:
    @staticmethod
    def cosine_builtin(e1, e2):
        return 1 - spatial.distance.cosine(e1, e2)

    @staticmethod
    def pearson(e1, e2):
        return stats.pearsonr(e1, e1)[0]

    @staticmethod
    def spearman_rank(e1, e2):
        return stats.spearmanr(e1, e2)[0]

    @staticmethod
    def euclidean(e1, e2):
        return 1 - spatial.distance.euclidean(e1, e2)

    @staticmethod
    def cosine(e1, e2):
        nom = np.dot(e1, e2)

        e1_norm = np.sqrt(np.sum(e1**2))
        e2_norm = np.sqrt(np.sum(e2**2))

        denom = e1_norm*e2_norm
        return nom/denom