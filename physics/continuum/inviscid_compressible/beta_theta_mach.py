import numpy
from numpy import sin, cos, tan
from scipy.optimize import fsolve
from matplotlib import pyplot as plt


gamma = 1.4
half_gamma_minus = (gamma - 1) / 2


class BetaThetaMach(object):

    def __init__(self, mach_1):
        self._mach_1 = mach_1
        self._mach_1_square_inv = 1 / mach_1 / mach_1

    def theta(self, beta):
        sin_beta_square = sin(beta) * sin(beta)
        tan_theta = 2 * (
            (sin_beta_square - self._mach_1_square_inv) /
            (gamma + cos(2*beta) + 2*self._mach_1_square_inv) / tan(beta)
        )
        return numpy.arctan(tan_theta)

    def mach_2(self, beta):
        theta = self.theta(beta)
        mach_n_1 = self._mach_1 * sin(beta)
        mach_n_1_square = mach_n_1 * mach_n_1
        mach_n_2_square = ((1 + half_gamma_minus * mach_n_1_square) / 
                           (gamma * mach_n_1_square - half_gamma_minus))
        return numpy.sqrt(mach_n_2_square) / sin(beta - theta)


if __name__ == '__main__':
    # set up plot
    plt.figure(figsize=(8,4))
    plt.axis(xmin=0, xmax=50, ymin=0, ymax=90)
    plt.ylabel(r'$\beta\,(\deg)$')
    plt.xlabel(r'$\theta\,(\deg)$')
    plt.grid()
    # wave angle
    beta_deg = numpy.linspace(start=0.1, stop=89.9, num=101)
    beta_rad = numpy.deg2rad(beta_deg)
    theta_rad = numpy.zeros(len(beta_rad))
    mach = (1.2, 1.6, 2, 4, 8, 1024)
    for m in mach:
        shock = BetaThetaMach(m)
        i = 0
        for b in beta_rad:
            theta_rad[i] = shock.theta(b)
            i += 1
        plt.plot(numpy.rad2deg(theta_rad), beta_deg,
                 label=r'$M_\mathrm{Before}=$'+str(m))
    # sonic line
    beta_deg = list()
    theta_deg = list()
    mach_mid = 2.01
    mach = numpy.concatenate((numpy.linspace(start=1.01, stop=mach_mid),
                              numpy.linspace(start=mach_mid, stop=32)))
    for m in mach:
        shock = BetaThetaMach(m)
        func = lambda beta : shock.mach_2(beta) - 1
        beta = fsolve(func, numpy.deg2rad(30))[0]
        theta = shock.theta(beta)
        beta_deg.append(numpy.rad2deg(beta))
        theta_deg.append(numpy.rad2deg(theta))
    plt.plot(theta_deg, beta_deg, '--', label=r'$M_\mathrm{After}=1$')
    # output
    plt.legend()
    # plt.show()
    plt.savefig(fname='beta_theta_mach.pdf', format='pdf')
    