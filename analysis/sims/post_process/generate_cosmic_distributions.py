import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    From arXiv:1606.06907 [hep-ph] 2016
    """
    # generate arrays in MeV
    E_exponent = np.arange(1024)*(3.0/1024.0)+3.0
    Es = np.array([10.0**i for i in E_exponent])
    thetas = np.arange(1024)*np.pi/2.0/1024.0

    E_0 = 4.28*1e3
    epsilon = 854.0*1e3
    n = 3.0
    n_theta = 2.0

    def spectrum(E):
        return 1e12*(E_0+E)**(-n)/(1.0+E/epsilon)

    def divergence(theta):
        return np.cos(theta)**n_theta

    # generate file 1
    with open('muon_spectrum.mac','w') as f:
        for e in Es:
            f.write('/gps/hist/point '+str(e)+' '+'%0.10f'%spectrum(e)+'\n')

    # check spectra
    # plt.plot(Es, spectrum(Es), ".")
    # plt.show()

    # generate file 2
    with open('muon_angular.mac','w') as f:
        for t in thetas:
            f.write('/gps/hist/point '+str(t)+' '+str(divergence(t))+'\n')

if __name__=="__main__":
    main()

