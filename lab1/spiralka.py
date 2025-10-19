import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

df = pd.read_csv("spiralka.csv", header = None)
df.columns = ["X", "Y"]
df["Y2"] = df["Y"] * 3

# spiralka

plt.plot(df["X"], df["Y"], marker = "s", linestyle="", color="black", markerfacecolor = 'none', markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# wykres 1

plt.scatter(df["X"], df["Y"], marker = "o", linestyle="", edgecolors="black", facecolors = 'none', s = 10)
plt.scatter(df["X"], df["Y2"], marker = "o", linestyle="", edgecolors="black", facecolors = 'none', s = 10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# wykres 2
x1 = [1,2,3,4,5]
y1 = [ 2,4,6,8,10]
y2 = [10,7,5,3,2]

plt.figure(figsize=(6,4))
plt.plot(x1,y1)
plt.plot(x1,y2)
plt.show()

# wykres 3

# wykres 4
y4 = [ 3,1,4,1,5,9]
plt.figure(figsize=(6,4))
plt.plot(y4)
plt.show()