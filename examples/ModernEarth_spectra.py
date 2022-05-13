from rfast import Rfast
from matplotlib import pyplot as plt

r = Rfast('ModernEarth/rfast_inputs.scr')
F1, F2 = r.genspec_scr()
F1_noH2O, F2_noH2O = r.genspec_scr(omit_gases=['h2o'])

plt.rcParams.update({'font.size': 18})
fig,ax = plt.subplots(1,1,figsize=[8,5])

ax.plot(r.lam,F2,drawstyle='steps-mid',c='k', lw=.5)
ax.fill_between(r.lam, F2, F2_noH2O, lw=0, step='mid', alpha=.3, label='H$_2$O')

ax.set_ylabel('Planet-to-Star flux ratio')
ax.set_xlabel('Wavelength ($\mu$m)')
ax.legend(ncol=1,bbox_to_anchor=(0.98,0.98),loc='upper right')
ax.grid()

plt.show()