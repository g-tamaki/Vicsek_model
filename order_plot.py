import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [0.0, 0.9998309378153204],
    [0.1, 0.971666862511216],
    [0.2, 0.9001647603014683],
    [0.3, 0.7963392365056187],
    [0.4, 0.6408669574596928],
    [0.5, 0.3076182810920998],
    [0.6, 0.08008597791291498],
    [0.7, 0.026328736364920346],
    [0.8, 0.004054810190902796],
    [0.9, 0.002602742712273006],
    [1.0, 0.001018604861399327],
])
plt.plot(data[:, 0], data[:, 1], marker="o")
# 軸を描く
plt.
plt.xlabel("Noise intensity: eta")
plt.ylabel("abs(mf)")
plt.show()