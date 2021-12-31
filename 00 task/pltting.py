import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(6.0, 1.0, 100000)
plt.hist(x, bins= 100)
plt.show()