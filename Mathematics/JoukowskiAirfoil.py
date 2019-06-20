import numpy as np
from matplotlib import pyplot as plt

chord = 1
gamma = 0.2
beta = 10/180 * np.pi

def joukowski(z):
  return (z + chord**2 / z) / 2

# circle in z-plane
radius = (gamma+1) * chord / np.cos(beta)
center = -gamma + 1j*(gamma+1) * np.tan(beta)
center *= chord
theta = np.linspace(start=0, stop=2 * np.pi)

# create data
curves_in_z = list()
curves_in_w = list()
s_min, s_max = 1.0, 2.5
# circles
for scale in np.linspace(start=s_min, stop=s_max, num=4):
  z = center + scale * radius * np.exp(1j*theta)
  curves_in_z.append(z)
  curves_in_w.append(joukowski(z))
# rays
rho = radius * np.linspace(s_min, s_max)
for theta in np.linspace(start=0, stop=2*np.pi, num=12, endpoint=False)-beta:
  z = center + rho * np.exp(1j*theta)
  curves_in_z.append(z)
  curves_in_w.append(joukowski(z))

# plot
plt.figure(figsize=(8,4))
# z-plane
plt.subplot(1, 2, 1)
plt.grid(True)
plt.axis('equal')
for chord in curves_in_z:
  if chord is curves_in_z[0]:
    plt.plot(chord.real, chord.imag, 'r')
  else:
    plt.plot(chord.real, chord.imag, 'b--')
# w-plane
plt.subplot(1, 2, 2)
plt.grid(True)
plt.axis('equal')
for a in curves_in_w:
  if a is curves_in_w[0]:
    plt.plot(a.real, a.imag, 'r')
  else:
    plt.plot(a.real, a.imag, 'b--')
# write
plt.tight_layout()
plt.savefig('JoukowskiAirfoil.pdf')
