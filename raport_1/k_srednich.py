import numpy as np
import math
import matplotlib.pyplot as plt

def f_euklidesowa(pktA, pktB):
    return math.sqrt((pktA[0] - pktB[0]) ** 2 + (pktA[1] - pktB[1]) ** 2)

def f_odleglosc(pktA, pktB):
    return math.fabs(pktA[0] - pktB[0])

def k_srednich(dane, iters, m, miara_odleglosci, iteracje_wizualizacji=[]):
    dane = np.array(dane)
    n_probek, n_cech = dane.shape

    indeksy_poczatkowe = np.random.choice(n_probek, m, replace=False)
    v = dane[indeksy_poczatkowe, :].astype(float)
    labels = np.zeros(n_probek, dtype=int)

    for i in range(iters):

        for s in range(n_probek):
            odleglosci = []
            for j in range(m):
                odleglosc = miara_odleglosci(dane[s], v[j])
                odleglosci.append(odleglosc)
            labels[s] = np.argmin(odleglosci)

        for j in range(m):
            probki_dla_grupy = dane[labels == j]
            if len(probki_dla_grupy) == 0:
                continue
            v[j] = np.mean(probki_dla_grupy, axis=0)


        if i+1 in iteracje_wizualizacji:
            print(f"\nIteracja {i+1}")
            for j in range(m):
                probki = dane[labels == j]
                if len(probki) == 0:
                    min_x1 = max_x1 = min_x2 = max_x2 = 0
                else:
                    min_x1, min_x2 = np.min(probki, axis=0)
                    max_x1, max_x2 = np.max(probki, axis=0)
                print(f"Grupa {j+1}:")
                print(f"  Środek: {v[j]}")
                print(f"  Liczba próbek: {len(probki)}")
                print(f"  x1: min={min_x1}, max={max_x1}")
                print(f"  x2: min={min_x2}, max={max_x2}")

            rysuj_wykres(dane, [(i+1, v.copy(), labels.copy(), m)], liczba_grup=m)




def rysuj_wykres(dane, historia, liczba_grup):
    kolory = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

    for rekord in historia:
        iteracja, srodki, labels, m = rekord
        plt.figure(figsize=(6,5))
        for j in range(liczba_grup):
            probki = dane[labels == j]
            kolor = kolory[j % len(kolory)]
            plt.scatter(probki[:,0], probki[:,1], color=kolor, label=f'Grupa {j+1}')
            plt.scatter(srodki[j,0], srodki[j,1], color=kolor, marker='X', s=200, edgecolor='k')
        plt.title(f'Iteracja {iteracja}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()



dane = np.loadtxt("../spiralka.csv", delimiter=',')


# RAPORT 1
k_srednich(dane, iters=10, m=3, miara_odleglosci=f_euklidesowa,iteracje_wizualizacji=[4,10])

# RAPORT 2
k_srednich(dane, iters=10, m=4, miara_odleglosci=f_odleglosc,iteracje_wizualizacji=[4, 10])

