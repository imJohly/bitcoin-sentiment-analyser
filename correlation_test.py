import numpy as np

import matplotlib.pyplot as plt

# time array
t = np.arange(0, 10, 0.1)

# Generate signals
signal1 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.cos(2 * np.pi * 3 * t) + np.random.normal(0, 0.1, len(t))
signal2 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.cos(2 * np.pi * 3 * t) + np.random.normal(0, 0.1, len(t))

numpy_correlation = np.corrcoef(signal1, signal2)[0, 1]
print('NumPy Correlation:', numpy_correlation)

plt.plot(signal1)
plt.plot(signal2)
plt.show()

