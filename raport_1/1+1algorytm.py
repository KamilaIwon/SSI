import random
import numpy as np
import math
import matplotlib.pyplot as plt

def f_przystosowania(x):
    return math.sin(x/10) * math.sin(x/200)

def algorytm(rozrzut, wsp_przyrostu, l_iteracji, zakres_zmiennosci=None):
    if zakres_zmiennosci is None:
        zakres_zmiennosci = [0, 100]

    x = random.randint(zakres_zmiennosci[0], zakres_zmiennosci[1])
    y = f_przystosowania(x)
    dane_opisowe(-1,x,y,rozrzut)
    xs = []
    ys= []
    for i in range(l_iteracji):
        x_pot = x + random.uniform(-rozrzut, rozrzut)
        if x_pot < zakres_zmiennosci[0]:
            x_pot = zakres_zmiennosci[0]
        elif x_pot>zakres_zmiennosci[1]:
            x_pot=zakres_zmiennosci[1]
        y_pot = f_przystosowania(x_pot)
        if(y_pot>=y):
            x = x_pot
            y= y_pot
            rozrzut *= wsp_przyrostu
        else:
            rozrzut/=wsp_przyrostu

        xs.append(x)
        ys.append(y)
        dane_opisowe(i, x, y, rozrzut)
    rysuj(xs,ys)

def dane_opisowe(iter, x, y, rozrzut):
    print(f"iteracja {iter+1}:  x={x}, y={y}, rozrzut= {rozrzut}")

def rysuj(xs,ys,zakres_zmiennosci=None):
    if zakres_zmiennosci is None:
        zakres_zmiennosci = [0, 100]
    x_lin = np.linspace(zakres_zmiennosci[0], zakres_zmiennosci[-1], 1000)
    y_lin = [f_przystosowania(x) for x in x_lin]
    plt.figure(figsize=(10,5))
    plt.plot(x_lin,y_lin,label="y=sin(x/10)*sin(x/200)", linewidth=2)
    plt.scatter(xs,ys,color="red", s=20, label="punkty algorytmu")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Algorytm 1+1 z funkcją przystosowania y=sin(x/10)*sin(x/200)")
    plt.legend()
    plt.grid()
    plt.show()



algorytm(10, 1.1, 20)