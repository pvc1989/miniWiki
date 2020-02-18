import numpy as np
from scipy.fftpack import fft, ifft


def truncate(u_sample, x_output, n_terms):
  u_hat = fft(u_sample)
  n = len(x_output) - 1
  w = np.exp(np.pi * 2.j / n)
  x = x_output - x_output[0]
  x /= x[-1]
  x *= n
  u_sum = np.real(u_hat[0]) * np.ones(n + 1)
  for k in range(1, n_terms):
    lanczos_k = np.sinc(2*k/n)
    u_sum += np.real(u_hat[k] * (w ** (k * x))) * 2 * lanczos_k
  return u_sum / len(u_sample)


if __name__ == "__main__":
  u_exact = lambda x : np.sign(x)
  x_sample = np.arange(start=-1.0, stop=1.0, step=0.001)
  u_sample = u_exact(x_sample)
  from matplotlib import pyplot as plt
  plt.plot(x_sample, u_sample, 'r.')
  x_output = np.linspace(start=-1.0, stop=1.0, num=501)
  plt.plot(x_output, truncate(u_sample, x_output, n_terms=40), 'b-')
  plt.show()
