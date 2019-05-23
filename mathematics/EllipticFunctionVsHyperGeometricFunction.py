import numpy as np
from scipy import special
from matplotlib import pyplot as plt

# ellipk(1.0) goes to inf, so choose stop < 1.0
x = np.linspace(start=0, stop=0.8)
# ellipk(m) = \int_0^{\pi/2} \frac{dt}{\sqrt{1 - m \sin^2 t}}, in which, m = k^2
plt.plot(x, special.ellipk(x), 'b-')
plt.plot(x, special.hyp2f1(1/2, 1/2, 1, x)*np.pi/2, 'ro')
plt.show()
# ellipe(m) = \int_0^{\pi/2} \sqrt{1 - m \sin^2 t} dt, in which, m = k^2
plt.plot(x, special.ellipe(x), 'b-')
plt.plot(x, special.hyp2f1(1/2, -1/2, 1, x)*np.pi/2, 'ro')
plt.show()
