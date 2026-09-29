from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-8, 8, 400)
plt.plot(z, 1 / (1 + np.exp(-z)))
plt.axhline(0.5, linestyle="--")
plt.axvline(0, linestyle="--")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")

output_path = Path(__file__).resolve().parent / "sigmoid.png"
plt.savefig(output_path)
plt.close()