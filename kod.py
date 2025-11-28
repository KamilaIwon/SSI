import matplotlib
matplotlib.use("Agg")   # ważne, bo PDF nie wymaga okna
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import matplotlib.pyplot as plt
import csv


# ----------------------------------------------------
# Wczytanie danych z CSV
# ----------------------------------------------------
data = []
with open("spiralka.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        data.append([float(row[0]), float(row[1])])

probki = np.array(data)
M = len(probki)
n = 2


# ----------------------------------------------------
# Funkcje odległości
# ----------------------------------------------------
def dist_euclid(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def dist_x1(a, b):
    return abs(a[0] - b[0])


# ----------------------------------------------------
# Rysowanie wykresu
# ----------------------------------------------------
def rysuj(probki, centers, groups, opis):
    fig = plt.figure(figsize=(7, 7))
    m = len(centers)

    colors = ["red", "blue", "green", "orange", "purple"]

    for j in range(m):
        Xgr = probki[groups == j]

        plt.scatter(Xgr[:, 0], Xgr[:, 1], c=colors[j], alpha=0.7)
        plt.scatter(
            centers[j, 0], centers[j, 1],
            c=colors[j], marker="X", s=250,
            edgecolors="black", linewidths=1.5
        )

    plt.title(opis)
    plt.grid(True)

    return fig


# ----------------------------------------------------
# Funkcja raportowa – tekst
# ----------------------------------------------------
def raport_tekst(pdf, centers, groups, probki, opis):
    fig = plt.figure(figsize=(8.5, 11))  # strona A4 pion
    plt.axis('off')

    text = f"{opis}\n\n"
    m = len(centers)

    for j in range(m):
        Xgr = probki[groups == j]
        text += f"--- Grupa {j + 1} ---\n"
        text += f"Środek: {centers[j]}\n"
        text += f"Liczba próbek: {len(Xgr)}\n"
        if len(Xgr) > 0:
            text += f"Min x1: {np.min(Xgr[:, 0])}   Max x1: {np.max(Xgr[:, 0])}\n"
            text += f"Min x2: {np.min(Xgr[:, 1])}   Max x2: {np.max(Xgr[:, 1])}\n"
        text += "\n"

    plt.text(0.01, 0.99, text, fontsize=10, va="top", family="monospace")

    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------
# K-means zwracający wyniki z iteracji 4 i 10
# ----------------------------------------------------
def k_means(probki, m, iters, dist_func):
    "np.random.seed(123)"

    idx = np.random.choice(len(probki), m, replace=False)
    centers = probki[idx].copy()
    groups = np.zeros(len(probki), dtype=int)

    centers4 = None
    groups4 = None

    for it in range(1, iters + 1):

        # przypisanie próbek
        for s in range(M):
            d = [dist_func(probki[s], centers[j]) for j in range(m)]
            groups[s] = np.argmin(d)

        # zapis po 4 iteracjach
        if it == 4:
            centers4 = centers.copy()
            groups4 = groups.copy()

        # aktualizacja środków
        for j in range(m):
            Xgr = probki[groups == j]
            if len(Xgr) > 0:
                centers[j] = np.mean(Xgr, axis=0)

    # zwrot: po 4 i po 10
    return centers4, groups4, centers.copy(), groups.copy()


def fcm(probki, m, iters, fcm_m=2):
    "np.random.seed(123)"

    M = len(probki)
    n = probki.shape[1]

    # ------------------------------------------------------------
    # 1. INICJALIZACJA
    # ------------------------------------------------------------
    # losowe U (przynależności)
    U = np.random.rand(m, M)
    U = U / np.sum(U, axis=0)   # normalizacja – suma przynależności = 1

    # obliczamy początkowe środki
    V = np.zeros((m, n))
    for j in range(m):
        um = U[j] ** fcm_m
        V[j] = np.sum(um[:, None] * probki, axis=0) / np.sum(um)

    # kopie wyników dla WYMAGANYCH iteracji
    V4 = None
    U4 = None

    # ------------------------------------------------------------
    # 2. PĘTLA GŁÓWNA
    # ------------------------------------------------------------
    for it in range(1, iters + 1):

        # 2.1 Odległości D
        D = np.zeros((m, M))
        for j in range(m):
            for s in range(M):
                D[j, s] = np.linalg.norm(probki[s] - V[j])

        # 2.2 Minimalna wartość w D
        D[D < 1e-5] = 1e-5

        # 2.3 Aktualizacja U
        for j in range(m):
            for s in range(M):
                denom = np.sum((D[j, s] / D[:, s]) ** (2 / (fcm_m - 1)))
                U[j, s] = 1.0 / denom

        # 2.4 Sprawdzanie NaN
        if np.isnan(U).any():
            print("Błąd: W macierzy U pojawiło się NaN!")
            return None

        # 2.5 Aktualizacja środków V
        for j in range(m):
            um = U[j] ** fcm_m
            V[j] = np.sum(um[:, None] * probki, axis=0) / np.sum(um)

        # --------------------------------------------------------
        # Zapis wyników po 4 iteracjach
        # --------------------------------------------------------
        if it == 4:
            V4 = V.copy()
            U4 = U.copy()

    # zwrot: po 4 i po 20 iteracjach
    return V4, U4, V.copy(), U.copy()

def raport_tekst_fcm(pdf, V, U, probki, opis):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')

    text = f"{opis}\n\n"
    m = len(V)

    for j in range(m):
        # wybieramy próbki z U > 0.6
        idx = np.where(U[j] > 0.6)[0]
        Xgr = probki[idx]

        text += f"--- Grupa {j + 1} ---\n"
        text += f"Środek: {V[j]}\n"
        text += f"Liczba próbek z U>0.6: {len(idx)}\n"

        if len(Xgr) > 0:
            text += f"Min x1: {np.min(Xgr[:,0])}   Max x1: {np.max(Xgr[:,0])}\n"
            text += f"Min x2: {np.min(Xgr[:,1])}   Max x2: {np.max(Xgr[:,1])}\n"

        text += "\n"

    plt.text(0.01, 0.99, text, fontsize=10, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def rysuj_fcm(probki, V, U, opis):
    fig = plt.figure(figsize=(7, 7))
    m = len(V)
    M = len(probki)

    # 1. Definiujemy kolory bazowe dla grup (RGB)
    # Grupa 1: Czerwony, Grupa 2: Zielony, Grupa 3: Niebieski
    # Jeśli m > 3, dodajemy kolejne (np. Cyjan, Magenta)
    base_colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 0.0, 1.0],  # Magenta (dla 4 grupy)
        [0.0, 1.0, 1.0],  # Cyan (dla 5 grupy)
    ])

    # Zabezpieczenie, gdyby m było większe niż zdefiniowane kolory
    if m > len(base_colors):
        print("Uwaga: Zdefiniuj więcej kolorów bazowych w funkcji rysuj_fcm!")
        base_colors = np.random.rand(m, 3)

    # Używamy tylko tylu kolorów, ile jest grup
    current_colors = base_colors[:m]

    # 2. Obliczamy kolor każdego punktu (Mieszanie)
    # Wynik to macierz (M, 3) zawierająca wartości R, G, B dla każdego punktu
    # Wzór: Kolor_Punktu = U[0]*Kolor_0 + U[1]*Kolor_1 + ...
    point_colors = np.dot(U.T, current_colors)

    # Upewniamy się, że wartości są w przedziale [0, 1] (powinny być, bo suma U=1)
    point_colors = np.clip(point_colors, 0, 1)

    # 3. Rysujemy wszystkie punkty naraz z wyliczonymi kolorami
    plt.scatter(probki[:, 0], probki[:, 1], c=point_colors, s=30)

    # 4. Rysujemy środki grup (tym razem "czystym" kolorem bazowym)
    for j in range(m):
        plt.scatter(V[j, 0], V[j, 1],
                    c=[current_colors[j]],  # musi być w liście
                    marker="X", s=250,
                    edgecolors="black", linewidths=1.5,
                    label=f'Środek gr. {j + 1}')

    plt.title(opis)
    plt.grid(True)
    # plt.legend() # Opcjonalnie, ale przy mixie kolorów legenda jest mniej czytelna

    return fig


# ================================================================
# GENEROWANIE PDF
# ================================================================
with PdfPages("raport.pdf") as pdf:

    # ============================================================
    # 1. RAPORT – euklidesowa, 3 grupy
    # ============================================================
    c4, g4, c10, g10 = k_means(probki, m=3, iters=10, dist_func=dist_euclid)

    # --- tekst po 4 iteracjach
    raport_tekst(pdf, c4, g4, probki, "RAPORT 1 – po 4 iteracjach (odległość euklidesowa)")

    # --- wykres po 4 iteracjach
    fig = rysuj(probki, c4, g4, "Grupy po 4 iteracjach – euklidesowa")
    pdf.savefig(fig)
    plt.close(fig)

    # --- tekst po 10 iteracjach
    raport_tekst(pdf, c10, g10, probki, "RAPORT 1 – po 10 iteracjach (odległość euklidesowa)")

    # --- wykres po 10 iteracjach
    fig = rysuj(probki, c10, g10, "Grupy po 10 iteracjach – euklidesowa")
    pdf.savefig(fig)
    plt.close(fig)

    # ============================================================
    # 2. RAPORT – odległość |x1A − x1B|
    # ============================================================
    c4x, g4x, c10x, g10x = k_means(probki, m=4, iters=10, dist_func=dist_x1)

    # --- tekst po 10 iteracjach
    raport_tekst(pdf, c10x, g10x, probki, "RAPORT 2 – po 10 iteracjach (odległość |x1A − x1B|)")

    # --- wykres po 4 iteracjach
    fig = rysuj(probki, c4x, g4x, "Grupy po 4 iteracjach – odległość |x1A − x1B|")
    pdf.savefig(fig)
    plt.close(fig)

    # ============================================================
    # 3. RAPORT – FCM
    # ============================================================
    V4, U4, V20, U20 = fcm(probki, m=3, iters=20, fcm_m=2)

    # --- tekst po 4 iteracjach
    raport_tekst_fcm(pdf, V4, U4, probki,
                     "RAPORT 3 – FCM – po 4 iteracjach (U>0.6)")

    # --- wykres po 4 iteracjach
    fig = rysuj_fcm(probki, V4, U4, "FCM – po 4 iteracjach")
    pdf.savefig(fig)
    plt.close(fig)

    # --- tekst po 20 iteracjach
    raport_tekst_fcm(pdf, V20, U20, probki,
                     "RAPORT 3 – FCM – po 20 iteracjach (U>0.6)")

    # --- wykres po 20 iteracjach
    fig = rysuj_fcm(probki, V20, U20, "FCM – po 20 iteracjach")
    pdf.savefig(fig)
    plt.close(fig)

print("\n>>> Gotowe! Plik zapisano jako raport.pdf\n")
