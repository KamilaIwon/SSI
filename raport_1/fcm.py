import numpy as np
import math
import matplotlib.pyplot as plt

def f_euklidesowa(pktA, pktB):
    return math.sqrt((pktA[0] - pktB[0])**2 + (pktA[1] - pktB[1])**2)

def fuzzy_c_means(dane, m, iters=10, fcm_m=2, miara_odleglosci=f_euklidesowa, iteracje_wizualizacji=[]):
    dane = np.array(dane)
    M, n_cech = dane.shape
    przynaleznosc = 0.6

    indeksy_poczatkowe = np.random.choice(M, m, replace=False)
    V = dane[indeksy_poczatkowe, :].astype(float)


    U = np.random.rand(m, M)
    U = U / np.sum(U, axis=0)

    eps = 1e-5

    for iteracja in range(iters):
        D = np.zeros((m, M))
        for j in range(m):
            for s in range(M):
                D[j, s] = miara_odleglosci(dane[s], V[j])
                if D[j, s] < eps:
                    D[j, s] = eps

        for j in range(m):
            for s in range(M):
                denom = 0
                for k in range(m):
                    denom += (D[j, s] / D[k, s])**(2/(fcm_m - 1))
                U[j, s] = 1 / denom

        if np.any(np.isnan(U)):
            raise ValueError("Wystąpiły nieokreślone wartości w macierzy przynależności U!")

        for j in range(m):
            for i in range(n_cech):
                licznik = np.sum((U[j, :]**fcm_m) * dane[:, i])
                mianownik = np.sum(U[j, :]**fcm_m)
                V[j, i] = licznik / mianownik

        if iteracja+1 in iteracje_wizualizacji:
            print(f"\nIteracja {iteracja+1}")
            for j in range(m):
                probki_filtr = dane[U[j, :] > przynaleznosc]
                liczba_prob = len(probki_filtr)

                if liczba_prob == 0:
                    min_x1 = max_x1 = min_x2 = max_x2 = 0
                else:
                    min_x1, min_x2 = np.min(probki_filtr, axis=0)
                    max_x1, max_x2 = np.max(probki_filtr, axis=0)

                print(f"Grupa {j + 1}:")
                print(f"  Środek: {V[j]}")
                print(f"  Liczba próbek z U>{przynaleznosc}: {liczba_prob}")
                print(f"  x1: min={min_x1}, max={max_x1}")
                print(f"  x2: min={min_x2}, max={max_x2}")

            rysuj_wykres_fcm(dane, V, U, m, iteracja+1)

    return V, U


def rysuj_wykres_fcm(dane, V, U, m, iteracja):
    base_colors = np.array([[1,0,0], [0,0,1], [0,1,0], [1,0.5,0], [0.5,0,1], [0,1,1]])
    kolory = [base_colors[i % len(base_colors)] for i in range(m)]

    plt.figure(figsize=(6,5))

    for s in range(dane.shape[0]):
        kolor = np.zeros(3)
        for j in range(m):
            kolor += U[j, s] * kolory[j]
        plt.scatter(dane[s,0], dane[s,1], color=kolor)

    for j in range(m):
        plt.scatter(V[j,0], V[j,1], color=kolory[j], marker='X', s=200, edgecolor='k')

    plt.title(f"Iteracja {iteracja} - FCM")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


dane = np.loadtxt("../spiralka.csv", delimiter=',')
V, U = fuzzy_c_means(dane, m=3, iters=20, fcm_m=2, miara_odleglosci=f_euklidesowa, iteracje_wizualizacji=[4,20])

