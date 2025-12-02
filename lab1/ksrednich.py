import math

import numpy as np
import pandas
import matplotlib

data = np.loadtxt("../spiralka.csv", delimiter=',')


samples = data[np.random.choice(data.shape[0], 3, replace=False)]
V = [[]]

def odległość_euklidesowa(pktA, pktB):
    return math.sqrt((pktA[1]-pktA[0])**2+(pktB[1]-pktB[0])**2)

a = [5,3]
b=[5,2]
print(odległość_euklidesowa(a,b))

def ksrednich(M, m, iters, odległość):
    samples = data[np.random.choice(M.shape[0], m, replace=False)]
    for iter in range(iters):
        for s in range(M):
            u = []
            groups = []
            for V in samples:
                u.append(odległość(s,V))
            groups.append(u)
            X_gr = []
            for j in range(m):
                pass
                




