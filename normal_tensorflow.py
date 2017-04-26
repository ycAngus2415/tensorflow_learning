import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = 1/(np.sqrt(2*np.pi)*0.5)*np.exp(-np.power(x, 2)/(2*0.5*0.5))
plt.plot(x, y, 'ro')
plt.show()
