import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')

# for Joule running on Cori
import matplotlib
matplotlib.use('Agg')

# from 1169 <= cycle <= 1171, check_raw_spectrum
e_kev = [2614.5, 1460.8, 1120.3, 911.2, 609.3, 583.2]
# e_uncal = [6641.0, 3607, 2720, 2172, 1409, 1341]  # with 150 us pz const
# e_uncal = [7019.0, 3814, 2872, 2301, 1491, 1419] # 50 us pz const
e_uncal = [6975.0, 3785., 2850., 2285., 1479., 1410.] # 50 us pz const, 4-2-4 trap, run 110

# fit to 2nd order poly
poly = np.polyfit(e_kev, e_uncal, 2)
print(poly)
e_fit = np.poly1d(e_uncal)

plt.plot(e_kev, e_fit, '-r', lw=2, 
         label='p2={:.2e}  p1={:.2e}  p0={:.2e}'.format(*poly))
plt.plot(e_kev, e_uncal, '.k', ms=10)

plt.xlabel('E (keV)', ha='right', x=1)
plt.ylabel('E (uncal)', ha='right', y=1)

plt.legend()
# plt.show()
plt.savefig('./plots/energy_linearity.png')