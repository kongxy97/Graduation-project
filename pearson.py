import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class GetImportDots:
    def __init__(self):
        pass

    def pearson_affinity(self,M):
        return 1 - np.array([[pearsonr(a, b)[0] for a in M] for b in M])


    def weight_pearson_affinity(self,M):
        return np.array([[self.weight_pearson_distance(a, b) for a in M] for b in M])

    def weight_matrix(self,s1):
        epsonRate = 0.1
        line_range = (np.max(s1)-np.min(s1)) * epsonRate
        matrix = np.ones(len(s1))
        l1 = []
        for i in range(len(s1)-1):
            if 0 < i < len(s1):
                if (s1[i] >= s1[i-1] and s1[i] > s1[i+1]) or (s1[i] > s1[i-1] and s1[i] >= s1[i+1]) \
                        or (s1[i] <= s1[i-1] and s1[i] < s1[i+1]) or (s1[i] < s1[i-1] and s1[i] <= s1[i+1]):
                    if self.tmp_distance(s1[i-1], s1[i], s1[i+1]) > line_range:
                        matrix[i] = 2.0
                    else:
                        matrix[i] = 1.5
                    l1.append(i)

        for i in range(len(l1)):
            if 0 < i < len(l1):
                if (s1[i] >= s1[i-1] and s1[i] > s1[i+1]) or (s1[i] > s1[i-1] and s1[i] >= s1[i+1]) \
                        or (s1[i] <= s1[i-1] and s1[i] < s1[i+1]) or (s1[i] < s1[i-1] and s1[i] <= s1[i+1]):
                    if self.tmp_distance(s1[i - 1], s1[i], s1[i + 1]) > line_range:
                        matrix[i] = 2.0
                    else:
                        matrix[i] = 1.5
        return matrix

    def weight_pearson_distance(self,s1, s2):
        matrix_s1 = self.weight_matrix(s1)
        matrix_s2 = self.weight_matrix(s2)
        mDLP1 = self.DLP_distance(s1, matrix_s1)
        mDLP2 = self.DLP_distance(s2, matrix_s2)
        return 1 - self.cov_DLP_distance(s1, s2, mDLP1, mDLP2, matrix_s1, matrix_s2)/\
               np.sqrt(self.cov_DLP_distance(s1, s1, mDLP1, mDLP1, matrix_s1, matrix_s1) * self.cov_DLP_distance(s2, s2, mDLP2, mDLP2, matrix_s2, matrix_s2))

    def tmp_distance(self, a, b, c):
        return abs(a+ (a-c)/2 - b)

    def DLP_distance(self, s1, matrix_s1):
        sum1 = 0.0;
        sum2 = 0.0;
        for i in range(len(s1)):
            sum1 += matrix_s1[i] * s1[i];
            sum2 += matrix_s1[i];
        return sum1/sum2;

    def cov_DLP_distance(self,s1, s2, mDLP1, mDLP2, matrix_s1, matrix_s2):
        sum1 = 0.0;
        sum2 = 0.0;
        for i in range(len(s1)):
            sum1 += (s1[i] - mDLP1) * (s2[i] -mDLP2) * matrix_s1[i] * matrix_s2[i]
            sum2 += matrix_s1[i] * matrix_s2[i];
        return sum1 / sum2;