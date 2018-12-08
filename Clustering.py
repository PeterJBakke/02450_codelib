"""
Small functions for clustering exercises
"""
import numpy as np


class GMM_1d:
    def __init__(self, x):
        self.x = x

    def prob_nth_cluster(self, w, mu, sigma2, target, cluster):
        denom = 0
        for i in range(len(w)):
            temp = w[i] * (1 / (2 * np.pi * sigma2[i] )**0.5 ) * np.exp(- 1 / (2 * sigma2[i]) * (target - mu[i])**2)
            denom += temp

        num = w[cluster-1] * (1 / (2 * np.pi * sigma2[cluster-1] )**0.5 ) * np.exp(- 1 / (2 * sigma2[cluster-1]) * (target - mu[cluster-1])**2)

        return print('The probability that the observation belong to the {} cluster is: {}'.format(cluster, num / denom))



if __name__ == '__main__':
    data = [5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4]

    target  = data[7]
    cluster = 2
    w = [0.37, 0.29, 0.34]
    mu = [6.12, 6.55, 6.93]
    sigma2 = [0.09, 0.13, 0.12]

    myGMM = GMM_1d(x=data)

    myGMM.prob_nth_cluster(w=w, mu=mu, sigma2=sigma2, target=target, cluster=cluster)

